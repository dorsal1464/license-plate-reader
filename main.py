import argparse
import cv2
from plate_reader import PlateReader, VideoFeed


def main():
    parser = argparse.ArgumentParser(description="License Plate Reader runner")
    parser.add_argument('--ip', type=str, help='IP address', required=False)
    parser.add_argument('--filename', type=str, help='test runner on Filename', required=False, default=None)
    parser.add_argument('--mode', type=str, choices=['local', 'remote'], default="remote", help='Mode (either "local" or "remote" is accepted)')

    args = parser.parse_args()

    ip = args.ip
    filename = args.filename
    mode = args.mode
    plate_reader = PlateReader()

    if filename:
        frame = cv2.imread(filename)
        plate_reader.detect_plates(frame)
        VideoFeed("").display_frame(cv2.imencode('.jpg', frame)[1].tobytes())
        return

    elif mode == "remote":
        if not ip:
            print("ip parameter isn't set")
            return
        video_feed = VideoFeed(ip)
        # video_feed.display_local_video_feed(f"http://{ip}",plate_reader)
        video_feed.display_video_feed(plate_reader)
        return
    elif mode == "local":
        video_feed = VideoFeed("")
        if not ip or not ip.isalnum():
            ip = 1
        elif ip.isalnum():
            ip = int(ip)
        
        video_feed.display_local_video_feed(ip, plate_reader)
        return
    else:
        print(f"Unknown config: {args}")
    

if __name__ == "__main__":
    main()
