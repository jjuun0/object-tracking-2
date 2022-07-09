import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2
import csv
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import time

def img_Contrast(img):
    # 이미지 강조(대비)
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


def csv_write(save_path, bbox, save_name):
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f'{save_name}.csv'
    f = open(file_path, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(bbox)
    f.close()


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}/{}'.format(env.results_path, "LaSOT", self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)
            if frame_num == 80:
                print()

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        # cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break
        i = 0
        while True:
            i += 1
            print(f'{i}th frame start!')
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.imwrite(f'result/{i:03d}.png', frame_disp)
            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)
            #
            #     cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def draw_bboxes(self, img, tracker_num, bbox, color, identities=None, offset=(0, 0)):
        x, y, w, h = bbox
        label = str(tracker_num)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 5, 5)[0]
        # cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 6)
        cv2.rectangle(img, (x, y), (x + t_size[0], y + t_size[1]), color, -1)
        # cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        cv2.putText(img, label, (x, y + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 5, [255, 255, 255], 5)
        return img

    def run_focal(self, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        from FocalDataloader import LoadFocalFolder
        from config_parser import ConfigParser
        import torch
        from datetime import datetime
        import sys

        config = ConfigParser('./config.json')
        # exper_name = config['name']
        # check_iou = config['check_iou']
        write_bbox = config['write_bbox']
        write_conf = config['write_conf']
        write_time = config['write_time']
        is_record = config['is_record']
        video_name = config['video_name']
        # video_type = config['video_type']
        # img2d_ref = config['image2d_ref']
        start_frame_num = config['start_frame_num']
        last_frame_num = config['last_frame_num']
        start_focal_num = config['start_focal_num']
        last_focal_num = config['last_focal_num']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        save_path = Path(config['save_path']) / timestamp
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        save_path.mkdir(parents=True, exist_ok=True)
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        # multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'parallel'))
        #
        # if multiobj_mode == 'default':
        #     tracker = self.create_tracker(params)
        #
        # elif multiobj_mode == 'parallel':
        #     tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        # else:
        #     raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        # assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        # ", videofilepath must be a valid videofile"

        output_boxes = []

        cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
        # cv.resizeWindow(display_name, 960, 720)
        # success, frame = cap.read()
        # cv.imshow(display_name, frame)

        focaldataloader = LoadFocalFolder(video_name, 'focal', frame_range=(start_frame_num, last_frame_num),
                                         focal_range=(start_focal_num, last_focal_num))

        def _build_init_info(box):
            return {'init_bbox': box}

        # if success is not True:
        #     print("Read frame from {} failed.".format(videofilepath))
        #     exit(-1)
        # if optional_box is not None:
        #     assert isinstance(optional_box, (list, tuple))
        #     assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
        #     tracker.initialize(frame, _build_init_info(optional_box))
        #     output_boxes.append(optional_box)
        # else:
        #     while True:
        #         cv.waitKey()
        #         frame_disp = frame.copy()
        #
        #         cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
        #                    1.5, (0, 0, 0), 1)
        #
        #         x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
        #         init_state = [x, y, w, h]
        #         tracker.initialize(frame, _build_init_info(init_state))
        #         output_boxes.append(init_state)
        #         break

        # while True:
        # k = 0
        frame_idx = start_frame_num
        bbox_color = [[0, 0, 255], [0, 255, 0], [255, 0, 0]]


        # best_idx = [0]
        d_index = [5,5,5]
        is_first_frame = True
        print("Please type the number of trackers: ")
        tracker_num = int(sys.stdin.readline())
        tracker = []
        best_idx = [0 for _ in range(tracker_num)]
        for k in range(tracker_num):
            tracker.append(self.create_tracker(params))

        for frame, focals in focaldataloader:
            if frame_idx > start_frame_num:
                is_first_frame = False


            if is_first_frame:
                for k in range(tracker_num):
                    cv2.putText(frame, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                               1.5, (0, 0, 0), 1)

                    x, y, w, h = cv2.selectROI(video_name, frame, fromCenter=False)
                    init_state = [x, y, w, h]
                    tracker[k].initialize(frame, _build_init_info(init_state))
                    output_boxes.append(init_state)

            print(f'{frame_idx}th frame start!')
            if write_bbox:
                bbox_list = [frame_idx]
            if write_conf:
                conf_list = [frame_idx]
            if write_time:
                time_list = [frame_idx]
            for k in range(tracker_num):
                # print(f'{k+1}th tracking')
                if not is_first_frame:
                    tracker[k].first_track = False
                if write_time:
                    tracking_time_start = time.time()
                max_score = 0
                best_state = [0, 0, 0, 0]
                if is_first_frame:
                    range_focals = focals
                elif d_index[k] < best_idx[k] < last_focal_num - d_index[k]:
                    range_focals = focals[best_idx[k] - d_index[k]: best_idx[k] + d_index[k] + 1]
                    # Focal plane range 범위 벗어나지 않게 처리
                elif best_idx[k] <= d_index[k]:
                    range_focals = focals[0: d_index[k] * 2 + 1]
                else:
                    range_focals = focals[-(d_index[k] * 2 + 1):]

                # focal sequence
                for i, f in enumerate(range_focals):
                    if i == len(range_focals)-1:
                        tracker[k].last_sequence = True
                    i = int(f.split('\\')[-1][:3])

                    img = cv2.imread(f)
                    # # bright 밝게
                    # val = 10
                    # arr = np.full(img.shape, (val, val, val), dtype=np.uint8)
                    # res_img = cv2.add(img, arr)
                    # res_img = img_Contrast(img)


                    out = tracker[k].track(img)  # {"target_bbox": self.state, "conf_score": conf_score}
                    state = [int(s) for s in out['target_bbox']]
                    score = float(out['conf_score'])
                    print(f'[{k+1}th] focal {i} score: {score}')

                    if max_score < score:
                        max_score = score
                        output_boxes.append(state)
                        best_idx[k] = int(f.split('\\')[-1][:3])
                        best_state = state

                        # cv.rectangle(f, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                        #          (0, 255, 0), 5)

                    # font_color = (0, 0, 0)
                    # cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    #            font_color, 1)
                    # cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    #            font_color, 1)
                    # cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    #            font_color, 1)
                # cv2.rectangle(frame, (best_state[0], best_state[1]), (best_state[2] + best_state[0], best_state[3] + best_state[1]),
                #              bbox_color[k], 5)
                self.draw_bboxes(frame, k + 1, best_state, bbox_color[k])
                print(f'--[{k + 1}th] Best focal {best_idx[k]} score: {max_score}--')
                if write_bbox:
                    bbox_list.extend(best_state)
                if write_bbox:
                    conf_list.append(round(max_score, 2))
                if write_time:
                    tracking_time = time.time() - tracking_time_start
                    time_list.append(round(tracking_time, 2))
            if is_record:
                cv2.imwrite(f'{save_path}/{frame_idx:03d}.png', frame)

            if write_bbox:
                csv_write(save_path, bbox_list, 'bbox')
            if write_conf:
                csv_write(save_path, conf_list, 'conf')
            if write_time:
                csv_write(save_path, time_list, 'time')
            frame_idx += 1
            # print(f'best idx: {best_idx} focal num')
            cv2.imshow(video_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)
            #
            #     cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        # cap.release()
        cv2.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            # video_name = Path(videofilepath).stem
            base_results_path = 'result/result'

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def run_2d(self, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        from ImagesDataloader import Load2DFolder
        from config_parser import ConfigParser
        import torch
        from datetime import datetime
        import cv2

        config = ConfigParser('./config.json')
        exper_name = config['name']
        check_iou = config['check_iou']
        write_gt = config['write_gt']
        is_record = config['is_record']
        video_name = config['video_name']
        video_type = config['video_type']
        img2d_ref = config['image2d_ref']
        start_frame_num = config['start_frame_num']
        last_frame_num = config['last_frame_num']
        start_focal_num = config['start_focal_num']
        last_focal_num = config['last_focal_num']
        ckpt_path = config['pretrained_model']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        save_path = Path(config['save_path']) / timestamp
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        # assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        # ", videofilepath must be a valid videofile"

        output_boxes = []

        # cap = cv.VideoCapture(videofilepath)
        # display_name = 'Display: ' + tracker.params.tracker_name
        # cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv.resizeWindow(display_name, 960, 720)
        # success, frame = cap.read()
        # cv.imshow(display_name, frame)

        imagedataloader = Load2DFolder(video_name, frame_range=(start_frame_num, last_frame_num))

        def _build_init_info(box):
            return {'init_bbox': box}

        # if success is not True:
        #     print("Read frame from {} failed.".format(videofilepath))
        #     exit(-1)
        # if optional_box is not None:
        # #     assert isinstance(optional_box, (list, tuple))
        #     assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
        # #     tracker.initialize(frame, _build_init_info(optional_box))
        # #     output_boxes.append(optional_box)
        # else:
        #     while True:
        #         # cv.waitKey()
        #         # frame_disp = frame.copy()
        #
        #         # cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
        #         #            1.5, (0, 0, 0), 1)
        #
        #         x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
        #         init_state = [x, y, w, h]
        #         tracker.initialize(frame, _build_init_info(init_state))
        #         output_boxes.append(init_state)
        #         break

        # while True:
        k = 0
        frame_idx = start_frame_num
        max_score = 0
        best_state = None
        best_idx = [0]
        d_index = [5]
        is_first_frame = True
        for frame in imagedataloader:
            if is_first_frame:
                tracker.initialize(frame, _build_init_info(optional_box))
                is_first_frame = False
                continue

            print(f'{frame_idx}th frame start!')

            # Draw box

            out = tracker.track(frame) # {"target_bbox": self.state, "conf_score": conf_score}
            state = [int(s) for s in out['target_bbox']]
            score = int(out['conf_score'])
            print(f'score: {score}')

                    # cv.rectangle(f, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                    #          (0, 255, 0), 5)

                # font_color = (0, 0, 0)
                # cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #            font_color, 1)
                # cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #            font_color, 1)
                # cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                #            font_color, 1)
            cv.rectangle(frame, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)
            cv.imwrite(f'result/{frame_idx:03d}.png', frame)
            frame_idx += 1
            # print(f'best idx: {best_idx} focal num')
            # Display the resulting frame
            # cv.imshow(display_name, frame_disp)
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
            # elif key == ord('r'):
            #     ret, frame = cap.read()
            #     frame_disp = frame.copy()
            #
            #     cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
            #                (0, 0, 0), 1)
            #
            #     cv.imshow(display_name, frame_disp)
            #     x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
            #     init_state = [x, y, w, h]
            #     tracker.initialize(frame, _build_init_info(init_state))
            #     output_boxes.append(init_state)

        # When everything done, release the capture
        # cap.release()
        # cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")



