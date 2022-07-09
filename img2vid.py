import cv2
import os

def img2vid_plenoptic_total(root_folder, vid_name):
    frame_folders = []
    for frame in os.listdir(root_folder):
        if len(frame) == 3:
            frame_folders.append(os.path.join(root_folder, frame))

    imgs = []
    for frame_folder in frame_folders:
        for img in os.listdir(frame_folder):
            imgs.append(os.path.join(frame_folder, img))


    # imgs = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(img_folder, imgs[0]), cv2.IMREAD_COLOR)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 5.0
    video = cv2.VideoWriter(vid_name, fourcc, fps, (width, height))
    cnt = 0
    for img in imgs:
        print("...processing {}/{}".format(cnt + 1, len(imgs)))
        cnt += 1
        video.write(cv2.imread(img))
    cv2.destroyAllWindows()
    video.release()

def img2vid(img_folder, vid_name):
    # imgs = [img for img in os.listdir(img_folder) if img.endswith(".jpg")]
    imgs = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(img_folder, imgs[0]), cv2.IMREAD_COLOR)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 10.0
    video = cv2.VideoWriter(vid_name, fourcc, fps, (width, height))
    cnt = 0
    for img in imgs:
        video.write(cv2.imread(os.path.join(img_folder, img)))
        print("...processing {}/{}".format(cnt+1, len(imgs)))
        cnt += 1
    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    # 2D
    # img_folder = "D:/siame/NonVideo3/0502_104950"
    # vid_name = "D:/siame/NonVideo3/0502_104950.avi"

    img_folder = "D:/cswintt/newvideo1/0627_105535"
    vid_name = "D:/cswintt/newvideo1/0627_105535/result.avi"
    img2vid(img_folder, vid_name)

    # focal
    # img_folder = "D:/dataset/NonVideo3_tiny_result/nested"
    # vid_name = "D:/dataset/NonVideo3_tiny_result/nested_focal.avi"
    # img2vid_plenoptic_total(img_folder, vid_name)
