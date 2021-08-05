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
import time

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

imageFeed = 0

class MulticastServerProtocol:

    def __init__(self, show_feed=False):
        print("Launching Realsense Camera Server")
        self.t1_past = 0
        self.show_feed = show_feed

    def connection_made(self, transport):
        self.transport = transport


    def datagram_received(self, data, addr):
        global imageFeed
        self.workFlag = True
        self.transMode = struct.unpack('<h', data[:2]) # Mode
        self.depthScale = struct.unpack('<d', data[2:10]) # Scale
        self.timeStamp = struct.unpack('<d', data[10:18]) # TimeStamp
        #print(struct.unpack('<I', data[18:22])) # len
        depLen = struct.unpack('<I', data[18:22])# Image
        assert depLen != len(data[22:]), "Integrity Error"
        depImg = pickle.loads(data[22:])
        self.t1 = time.time()
        rateTime = self.t1 - self.t1_past
        
        print(f"Recieved from {addr}:{self.timeStamp} - {self.transMode} - {self.depthScale}")
        self.t1_past = self.t1
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depImg, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = cv2.resize(depth_colormap, dsize=None ,fx=3, fy=3)
        cv2.putText(depth_colormap, str(self.timeStamp), (0, 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(depth_colormap, str(round(rateTime, 3)) + "s - " + str(round(1/rateTime,2)) + "FPS", (0, 23),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        if self.show_feed:
            cv2.namedWindow(f'recieved-{addr}', cv2.WINDOW_NORMAL)
            cv2.imshow(f"recieved-{addr}", depth_colormap)
            cv2.waitKey(1)
        #print(depImg.shape)
        imageFeed = [addr, depth_colormap]
        #print('Received {!r} from {!r}'.format(data, addr))
        #data = "I received {!r}".format(data).encode("ascii")
        #print('Send {!r} to {!r}'.format(data, addr))
        #self.transport.sendto(data, addr)

def do_main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
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


if __name__ == "__main__":
    do_main()

