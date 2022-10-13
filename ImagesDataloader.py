import os
from PIL import Image
import numpy as np
import cv2


class Load2DFolder:
    def __init__(self, root, frame_range=None):
        self.root = root
        self.frames = [str(i) for i in range(frame_range[0], frame_range[1]+1)]
        self.frame_range = frame_range
        self.images_2d_path = []

        # frame range

        # self.frames = self.frames[self.frame_range[0]: self.frame_range[1] + 1]

        # 2D images setting
        for frame in self.frames:
            image_path = os.path.join(self.root, frame)
            image_path = os.path.join(image_path, 'images/007.png')
            # image_path = os.path.join(image_path, 'images/005.png')
            self.images_2d_path.append(image_path)

        # # focal images setting
        # for frame in self.frames:
        #     type_path = os.path.join(root, frame, self.type)
        #     focal_images_name = os.listdir(type_path)  # '000', '001',,,
        #     focal_planes = []
        #
        #     for focal_image in focal_images_name:
        #         focal_image_path = os.path.join(type_path, focal_image)
        #         focal_planes.append(focal_image_path)  # 'D:/dataset/NonVideo3_tiny\\000\\focal\\020.png',,
        #
        #     # focal range
        #     if self.focal_range is None:
        #         pass
        #     else:
        #         focal_planes = focal_planes[self.focal_range[0]: self.focal_range[1] + 1]
        #
        #     self.focal_images_path.append(focal_planes)
        #     # self.images_path.append(os.path.join(root, frame, self.type))

    def __getitem__(self, idx):
        path = self.images_2d_path[idx]
        img = cv2.imread(path)
        # focal_plane = self.focal_images_path[idx]
        # for focal_image_path in focal_plane:
        #     focal_image = cv2.imread(focal_image_path)
        #     focal_image_resize = focal_image_resize[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        #     focal_image_resize = np.ascontiguousarray(focal_image_resize)
        #     focal_planes_resize.append(focal_image_resize)

        # return img, focal_plane
        return img

        # 2D Images setting

    # def set_images_path(self):
    #     images_path = []
    #     for frame in self.frames:
    #         type_path = os.path.join(self.root, frame, 'images')
    #         image = os.listdir(type_path)[5]
    #         image_path = os.path.join(type_path, image)
    #         images_path.append(image_path)
    #     return images_path


if __name__ == '__main__':
    dataloader = Load2DFolder(root='/ssd2/vot_data/newvideo1/', frame_range=(65, 100))
    print(np.array(dataloader.images_2d_path).shape)
    img = dataloader[0]
    print(img)
