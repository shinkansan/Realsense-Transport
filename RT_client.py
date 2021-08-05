# RT (Realsense Transport) Client
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
import struct
import pickle
import logging

PORT = 1024
BROADCAST_ADDR = "239.255.255.250"

chunk_size = 4096

transmitMode = 2
# 0 for Depth, 1 for Color, 2 for Both color and depth

clipping_distance_in_meters = 12 #Clipping_in_meter


class realSense:

    def __init__(self, iscolor=True):
        self.pipeline, self.depth_scale = self.openPipeline()
        self.decimate_filter = rs.decimation_filter()
        self.decimate_filter.set_option(rs.option.filter_magnitude, 2)
        self.isColorMode = iscolor
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def openPipeline(self):
        print("Open RS Pipeline")
        pipeline = rs.pipeline()
        cfg = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = cfg.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        if device_product_line == 'L500':
            cfg.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        pipeline_profile = pipeline.start(cfg)
        sensor = pipeline_profile.get_device().first_depth_sensor()
        depth_scale = sensor.get_depth_scale()
        self.clipping_distance = clipping_distance_in_meters / depth_scale
        return pipeline, depth_scale

    def retrieveImage(self, pipeline):
        frames = pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        # take owner ship of the frame for further processing
        aligned_frames.keep()
        depth = aligned_frames.get_depth_frame()
        if self.isColorMode:
            color = aligned_frames.get_color_frame()
            #color = self.decimate_filter.process(depth)
        if depth:
            depth = self.decimate_filter.process(depth)
            # take owner ship of the frame for further processing
            depth.keep()
            # represent the frame as a numpy array
            depthData = depth.as_frame().get_data()        
            depthMat = np.asanyarray(depthData)
            grey_color = 127
            #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            depthMat = np.where((depthMat > self.clipping_distance) | (depthMat <= 0), grey_color, depthMat)
            depthMat = np.multiply(np.divide(depthMat,self.clipping_distance), 255).astype(np.uint8)
            depthMat = cv2.GaussianBlur(depthMat, (5,5), 0)
            ts = frames.get_timestamp()
            if self.isColorMode:
                colorMat = np.asanyarray(color.get_data())
                depthMat = cv2.resize(depthMat, dsize=(160,120))
                colorMat = cv2.resize(colorMat, dsize=(160,120))
                return depthMat, colorMat, ts
            return depthMat, None, ts
        else:
            return None, None, None


    def packetWriter(self):
        depth, color, timeStamp = self.retrieveImage(self.pipeline)

        if transmitMode == 0:

            raise NotImplementedError
        elif transmitMode == 1:

            raise NotImplementedError
        elif transmitMode == 2:
            #print(len(np.reshape(depth, (-1))))
            depthFrame = pickle.dumps(depth)
            colorFrame = pickle.dumps(color)
            data = b''.join([depthFrame, colorFrame])
            modeFrame = struct.pack('<h', transmitMode)
            scaleFrame = struct.pack('<d', self.depth_scale)
            tsFrame = struct.pack('<d', timeStamp)
            deplenFrame = struct.pack('<I', len(depthFrame))
            colorlenFrame = struct.pack('<I', len(colorFrame))

        self.debugViewer(depth, color)
        # Frame -> | MODE(2) | SCALE(8) | TIMESTAMP(8) | image(?) |
        frame_data = b''.join([modeFrame, scaleFrame, tsFrame, deplenFrame, depthFrame])
        #print(frame_data)
        return frame_data

    def debugViewer(self, depth, color=None):
        
        #grey_color = 153
        depth_image_3d = np.dstack((depth,depth,depth)) #depth image is 1 channel, color is 3 channels
        #bg_removed = np.where( (depth_image_3d <= 1), grey_color, color)
        #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color, depth_image_3d))

        cv2.namedWindow('send', cv2.WINDOW_NORMAL)
        cv2.imshow('send', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()


class DiscoveryClientProtocol:
    def __init__(self, loop, addr):
        self.loop = loop
        self.transport = None
        self.addr = addr

        self.rsHelper = realSense()

    def connection_made(self, transport):
        self.transport = transport
        sock = self.transport.get_extra_info('socket')

        sock.settimeout(3)
        addrinfo = socket.getaddrinfo(self.addr, None)[0]
        if addrinfo[0] == socket.AF_INET: # IPv4
            ttl = struct.pack('@i', 1)
            sock.setsockopt(socket.IPPROTO_IP, 
                socket.IP_MULTICAST_TTL, ttl)
        else:
            ttl = struct.pack('@i', 2)
            sock.setsockopt(socket.IPPROTO_IPV6, 
                socket.IPV6_MULTICAST_HOPS, ttl)

        print("Connected!")

    def datagram_received(self, data, addr):
        print("Reply from {}: {!r}".format(addr, data))
        # Don't close the socket as we might get multiple responses.

    def error_received(self, exc):
        print('Error received:', exc)

    def connection_lost(self, exc):
        print("Socket closed, stop the event loop")
        self.loop.stop()

    def sender(self):
        
        data = struct.pack("<d", 0.1235325)
        data2 = struct.pack("<d", 0.3215)
        data = b''.join([data, data2])
        data = self.rsHelper.packetWriter()
        #print(len(data))
        self.transport.sendto(data, (self.addr,PORT))
        
        

async def main_worker():
    loop = asyncio.get_event_loop()

    addrinfo = socket.getaddrinfo(BROADCAST_ADDR, None)[0]
    #print(addrinfo)
    sock = socket.socket(addrinfo[0], socket.SOCK_DGRAM) # UDP
    _, connect = await loop.create_datagram_endpoint(
        lambda: DiscoveryClientProtocol(loop,BROADCAST_ADDR),
        sock=sock,
    )
    #transport, protocol = loop.run_until_complete(connect)
    while True:
        connect.sender()
        await asyncio.sleep(1.0/30.0)

    loop.run_forever()
    transport.close()
    loop.close()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_worker())
    loop.close()
