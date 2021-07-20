# RT (Realsense Transport) Server
# though it's server, it recieve multicast image from clients (MEC)
# Author : Gwanjun, Shin
# Date : 2021.07.21.
# Acknowledgment : Fork from EtherSense(https://github.com/krejov100/EtherSense.git)

# Reinforce Depth Compression and Color+Depth Transportation Function

import pyrealsense2 as rs
import sys, getopt
import cv2
import lz4.frame
import asyncio
import numpy as np
import socket
if not hasattr(socket, 'SO_REUSEPORT'):
    socket.SO_REUSEPORT = 15
import struct
import pickle
import logging

BROADCAST_ADDR = "239.255.255.250"
PORT = 1024
chunk_size = 4096

transmitMode = 2
# 0 for Depth, 1 for Color, 2 for Both color and depth


class image_handler():
    
    def __init__(self, depth_scale):
        pass

    def show_color(self, frame):
        pass

    def show_depth(self):
        pass



class MulticastServerProtocol:

    def __init__(self):
        print("Launching Realsense Camera Server")

    def connection_made(self, transport):
        self.transport = transport


    def datagram_received(self, data, addr):
        print(struct.unpack('<h', data[:2])) # Mode
        print(struct.unpack('<d', data[2:10])) # Scale
        print(struct.unpack('<d', data[10:18])) # TimeStamp
        print(struct.unpack('<I', data[18:22])) # len
        depLen = struct.unpack('<I', data[18:22])# Image
        assert depLen != len(data[22:]), "Integrity Error"
        depImg = pickle.loads(data[22:])
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depImg, alpha=0.03), cv2.COLORMAP_JET)
        cv2.namedWindow('recieved', cv2.WINDOW_NORMAL)
        cv2.imshow("recieved", depth_colormap)
        cv2.waitKey(1)
        print(depImg.shape)
        #print('Received {!r} from {!r}'.format(data, addr))
        #data = "I received {!r}".format(data).encode("ascii")
        #print('Send {!r} to {!r}'.format(data, addr))
        #self.transport.sendto(data, addr)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    logging.basicConfig(level=logging.DEBUG)

    addrinfo = socket.getaddrinfo(BROADCAST_ADDR, None)[0]
    sock = socket.socket(addrinfo[0], socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    group_bin = socket.inet_pton(addrinfo[0], addrinfo[4][0])
    if addrinfo[0] == socket.AF_INET: # IPv4
        sock.bind(('', PORT))
        mreq = group_bin + struct.pack('=I', socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    else:
        sock.bind(('', PORT))
        mreq = group_bin + struct.pack('@I', 0)
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, mreq)



    listen = loop.create_datagram_endpoint(
        MulticastServerProtocol,
        sock=sock,
    )
    transport, protocol = loop.run_until_complete(listen)

    loop.run_forever()
    loop.close()