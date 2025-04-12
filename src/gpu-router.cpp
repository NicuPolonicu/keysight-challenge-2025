#include <array>
#include <vector>
#include <iostream>
#include <string>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <unistd.h>
#include <cstring>

#include <sycl/sycl.hpp>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/flow_graph.h>
#include "dpc_common.hpp"

#include <pcap.h>

const size_t burst_size = 32;
#define PACKET_SIZE 1518

struct Packet {
    std::vector<u_char> data;
    bool valid = false;
    std::array<u_char, 4> dest_ip{};
};

struct CountingStats {
    size_t ipv4 = 0;
    size_t ipv6 = 0;
    size_t icmp = 0;
    size_t udp = 0;
    size_t tcp = 0;
    size_t arp = 0;
};


const uint8_t IPV4_FLAG = 1<<1;
const uint8_t IPV6_FLAG = 1<<2;
const uint8_t ICMP_FLAG = 1<<3;
const uint8_t UDP_FLAG = 1<<4;
const uint8_t TCP_FLAG = 1<<5;
const uint8_t ARP_FLAG = 1<<6;

int main() {
    sycl::queue q;

    std::cout << "Using device: " <<
        q.get_device().get_info<sycl::info::device::name>() << std::endl;

    int nth = 10;
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nth);
    tbb::flow::graph g;

    int opened = 0;
    int expected_packets = 0;
    int nr_packets_read = 0;
    int packets_sent = 0;

    struct CountingStats final_stats = {0, 0, 0, 0, 0};

    tbb::flow::input_node<std::vector<Packet>> in_node{g,
        [&](tbb::flow_control& fc) -> std::vector<Packet> {
            static pcap_t* handle = nullptr;
            static char errbuf[PCAP_ERRBUF_SIZE];
            static const char* pcap_file = "../../src/capture2.pcap";

            if (!handle) {
                if (opened) {
                    std::cerr << "Error: pcap handle already opened" << std::endl;
                    std::cout << "Total packets read: " << nr_packets_read << "\n";
                    fc.stop();
                    return {};
                }
                handle = pcap_open_offline(pcap_file, errbuf);
                opened = 1;
                if (!handle) {
                    std::cerr << "Error opening pcap file: " << errbuf << std::endl;
                    fc.stop();
                    return {};
                }
            }

            std::vector<Packet> burst;
            burst.reserve(burst_size);

            for (size_t i = 0; i < burst_size; ++i) {
                struct pcap_pkthdr* header;
                const u_char* data;
                int result = pcap_next_ex(handle, &header, &data);

                if (result <= 0) {
                    pcap_close(handle);
                    handle = nullptr;
                    break;
                }

                nr_packets_read++;

                burst.push_back(Packet{std::vector<u_char>(data, data + header->caplen)});
            }

            if (burst.empty()) {
                std::cout << "No more packets to read\n";
                std::cout << "Total packets read: " << nr_packets_read << "\n";
                fc.stop();
            }

            return burst;
        }
    };

    tbb::flow::function_node<std::vector<Packet>, std::vector<Packet>> parse_packet_node {
        g, tbb::flow::unlimited, [&](std::vector<Packet> burst) {
            if (burst.empty()) return burst;
    
            const size_t N = burst.size();
            size_t total_size = 0;
    
            std::vector<size_t> packet_offsets(N);
            for (size_t i = 0; i < N; ++i) {
                packet_offsets[i] = total_size;
                total_size += burst[i].data.size();
            }
    
            std::vector<u_char> flat_data(total_size, 0);
            for (size_t i = 0; i < N; ++i) {
                std::memcpy(&flat_data[packet_offsets[i]], burst[i].data.data(), burst[i].data.size());
            }
    
            std::vector<uint8_t> valid_flags(N, 0);
            std::vector<std::array<u_char, 4>> dest_ips(N);
            struct CountingStats local_stats = {0, 0, 0, 0, 0};
            
    
            {
                sycl::buffer<u_char> buf(flat_data.data(), sycl::range<1>(flat_data.size()));
                sycl::buffer<size_t> offsets_buf(packet_offsets.data(), sycl::range<1>(N));
                sycl::buffer<uint8_t> valid_buf(valid_flags.data(), sycl::range<1>(N));
                sycl::buffer<std::array<u_char, 4>> ip_buf(dest_ips.data(), sycl::range<1>(N));
                sycl::buffer<CountingStats> stats_buf(&local_stats, sycl::range<1>(1));
    
                q.submit([&](sycl::handler& h) {
                    auto data = buf.get_access<sycl::access::mode::read>(h);
                    auto offsets = offsets_buf.get_access<sycl::access::mode::read>(h);
                    auto valid = valid_buf.get_access<sycl::access::mode::write>(h);
                    auto ips = ip_buf.get_access<sycl::access::mode::write>(h);
                    auto stats = stats_buf.get_access<sycl::access::mode::write>(h);
    
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                        size_t offset = offsets[i];
    
                        uint16_t eth_type = (data[offset + 12] << 8) | data[offset + 13];
                        // stats for IPV4, IPV6, ARP, ICMP, UDP, TCP
                        if (eth_type == 0x0800) {
                            valid[i] |= IPV4_FLAG;
                            uint8_t protocol = data[offset + 23];
                            if (protocol == 1) {
                                valid[i] |= ICMP_FLAG;
                            } else if (protocol == 6) {
                                valid[i] |= TCP_FLAG;
                            } else if (protocol == 17) {
                                valid[i] |= UDP_FLAG;
                            }
                        } else if (eth_type == 0x86DD) {
                            valid[i] |= IPV6_FLAG;
                            uint8_t protocol = data[offset + 20];
                            if (protocol == 58) {
                                valid[i] |= ICMP_FLAG;
                            } else if (protocol == 6) {
                                valid[i] |= TCP_FLAG;
                            } else if (protocol == 17) {
                                valid[i] |= UDP_FLAG;
                            }
                        } else if (eth_type == 0x0806) {
                            valid[i] |= ARP_FLAG;
                        }

                        if (valid[i] & IPV4_FLAG)
                            for (int j = 0; j < 4; ++j)
                                ips[i][j] = data[offset + 30 + j];
                    });
                }).wait_and_throw();
            }
    
            for (size_t i = 0; i < N; ++i) {
                if (valid_flags[i] & IPV4_FLAG) {
                    burst[i].valid = true;
                }
                burst[i].dest_ip = dest_ips[i];
                if (valid_flags[i] & IPV4_FLAG) local_stats.ipv4++;
                if (valid_flags[i] & IPV6_FLAG) local_stats.ipv6++;
                if (valid_flags[i] & ICMP_FLAG) local_stats.icmp++;
                if (valid_flags[i] & UDP_FLAG) local_stats.udp++;
                if (valid_flags[i] & TCP_FLAG) local_stats.tcp++;
                if (valid_flags[i] & ARP_FLAG) local_stats.arp++;
            }
            
            final_stats.ipv4 += local_stats.ipv4;
            final_stats.ipv6 += local_stats.ipv6;
            final_stats.icmp += local_stats.icmp;
            final_stats.udp += local_stats.udp;
            final_stats.tcp += local_stats.tcp;
            final_stats.arp += local_stats.arp;
            return burst;
        }
    };
    
    

    tbb::flow::function_node<std::vector<Packet>, std::vector<Packet>> routing_node {
        g, tbb::flow::unlimited, [&](std::vector<Packet> packets) {
            const size_t N = packets.size();
            size_t total_size = 0;
    
            std::vector<size_t> packet_offsets(N);
            for (size_t i = 0; i < N; ++i) {
                packet_offsets[i] = total_size;
                total_size += packets[i].data.size();
            }
    
            std::vector<u_char> flat_data(total_size, 0);
            for (size_t i = 0; i < N; ++i) {
                std::memcpy(&flat_data[packet_offsets[i]], packets[i].data.data(), packets[i].data.size());
            }
    
            {
                sycl::buffer<u_char> buf(flat_data.data(), sycl::range<1>(flat_data.size()));
                sycl::buffer<size_t> offsets_buf(packet_offsets.data(), sycl::range<1>(N));
    
                q.submit([&](sycl::handler& h) {
                    auto data = buf.get_access<sycl::access::mode::read_write>(h);
                    auto offsets = offsets_buf.get_access<sycl::access::mode::read>(h);
    
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                        size_t offset = offsets[i];
    
                        uint16_t eth_type = (data[offset + 12] << 8) | data[offset + 13];
                        if (eth_type != 0x0800) return;
    
                        for (int j = 30; j < 34; ++j)
                            data[offset + j] += 1;
                    });
                }).wait_and_throw();
            }
    
            for (size_t i = 0; i < N; ++i) {
                packets[i].data.assign(flat_data.begin() + packet_offsets[i],
                                       flat_data.begin() + packet_offsets[i] + packets[i].data.size());
    
                for (int j = 0; j < 4; ++j) {
                    packets[i].dest_ip[j] = packets[i].data[30 + j];
                }
            }
    
            return packets;
        }
    };
    

    
    

    tbb::flow::function_node<std::vector<Packet>, std::vector<Packet>> send_node {
        g, tbb::flow::unlimited, [&](std::vector<Packet> packets) {
            int sock = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
            if (sock < 0) {
                perror("Socket creation failed");
                return packets;
            }

            int one = 1;
            if (setsockopt(sock, IPPROTO_IP, IP_HDRINCL, &one, sizeof(one)) < 0) {
                perror("setsockopt failed");
                close(sock);
                return packets;
            }

            for (const auto& p : packets) {
                if (!p.valid || p.data.size() < 34) continue;
                packets_sent++;
                struct sockaddr_in dest{};
                dest.sin_family = AF_INET;
                dest.sin_port = 0;
                std::memcpy(&dest.sin_addr, p.dest_ip.data(), 4);

                // std::cout << "Sending to: "
                //           << (int)p.dest_ip[0] << "." << (int)p.dest_ip[1] << "."
                //           << (int)p.dest_ip[2] << "." << (int)p.dest_ip[3]
                //           << " size: " << p.data.size() << std::endl;

                ssize_t sent = sendto(sock, p.data.data(), p.data.size(), 0,
                                      (struct sockaddr*)&dest, sizeof(dest));
                if (sent < 0) perror("sendto failed");
            }

            close(sock);
            return packets;
        }
    };

    // Construct graph
    tbb::flow::make_edge(in_node, parse_packet_node);
    tbb::flow::make_edge(parse_packet_node, routing_node);
    tbb::flow::make_edge(routing_node, send_node);

    in_node.activate();
    g.wait_for_all();

    std::cout << "total packets read: " << nr_packets_read << "\n";
    std::cout << "total packets sent: " << packets_sent << "\n";
    
    std::cout << "Final stats:\n";
    std::cout << "IPV4: " << final_stats.ipv4 << "\n";
    std::cout << "IPV6: " << final_stats.ipv6 << "\n";
    std::cout << "ICMP: " << final_stats.icmp << "\n";
    std::cout << "UDP: " << final_stats.udp << "\n";
    std::cout << "TCP: " << final_stats.tcp << "\n";
    std::cout << "ARP: " << final_stats.arp << "\n";

    std::cout << "Done waiting" << std::endl;
}
