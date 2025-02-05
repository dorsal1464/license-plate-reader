import argparse
import cv2
import plate_reader


def main():
    parser = argparse.ArgumentParser(description="License Plate Reader runner")
    parser.add_argument('--ip', type=str, help='IP address', required=False)
    parser.add_argument('--filename', type=str, help='test runner on Filename', required=False, default=None)
    parser.add_argument('--mode', type=str, choices=['local', 'remote'], default="remote", help='Mode (either "local" or "remote" is accepted)')

    args = parser.parse_args()

    ip = args.ip
    filename = args.filename
    mode = args.mode

    if filename:
        frame = cv2.imread(filename)
        plate_reader.dark_magic_function2(frame)
        plate_reader.display_frame(cv2.imencode('.jpg', frame)[1].tobytes())
        return

    elif mode == "remote":
        plate_reader.display_video_feed(ip)
        return
    elif mode == "local":
        if ip and ip.isalnum():
            pass
        else:
            ip = 1
        plate_reader.display_local_video_feed(idx=ip)
        return
    else:
        print(f"Unknown config: {args}")
    

if __name__ == "__main__":
    main()
