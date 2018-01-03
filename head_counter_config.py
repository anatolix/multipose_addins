
from config import Configs, COCOSourceConfig
import numpy as np

class HeadCounterConfig:

    def __init__(self):
        self.width = 368
        self.height = 368

        self.stride = 8

        self.parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank",
                      "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear", "HeadCenter"]
        self.num_parts = len(self.parts)
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))
        self.parts += ["background"]
        self.num_parts_with_background = len(self.parts)

        leftParts, rightParts = self.ltr_parts(self.parts_dict)
        self.leftParts = leftParts
        self.rightParts = rightParts

        # this numbers probably copied from matlab they are 1.. based not 0.. based
        self.limb_from = ['neck', 'Rhip', 'Rkne', 'neck', 'Lhip', 'Lkne', 'neck', 'Rsho', 'Relb', 'Rsho', 'neck',
                          'Lsho', 'Lelb', 'Lsho',
                          'neck', 'nose', 'nose', 'Reye', 'Leye']
        self.limb_to = ['Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Rsho', 'Relb', 'Rwri', 'Rear', 'Lsho',
                        'Lelb', 'Lwri', 'Lear',
                        'nose', 'Reye', 'Leye', 'Rear', 'Lear']

        self.limb_from = [self.parts_dict[n] for n in self.limb_from]
        self.limb_to = [self.parts_dict[n] for n in self.limb_to]

        assert self.limb_from == [x - 1 for x in [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]]
        assert self.limb_to == [x - 1 for x in [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]]

        self.limbs_conn = list(zip(self.limb_from, self.limb_to))

        self.paf_layers = 2 * len(self.limbs_conn)
        self.heat_layers = self.num_parts
        self.num_layers = self.paf_layers + self.heat_layers + 1

        self.paf_start = 0
        self.heat_start = self.paf_layers
        self.bkg_start = self.paf_layers + self.heat_layers

        # self.data_shape = (self.height, self.width, 3)     # 368, 368, 3
        self.mask_shape = (self.height // self.stride, self.width // self.stride)  # 46, 46
        self.parts_shape = (self.height // self.stride, self.width // self.stride, self.num_layers)  # 46, 46, 57

        class TransformationParams:

            def __init__(self):
                self.target_dist = 0.6;
                self.scale_prob = 1;  # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
                self.scale_min = 0.5;
                self.scale_max = 1.1;
                self.max_rotate_degree = 40.
                self.center_perterb_max = 40.
                self.flip_prob = 0.5
                self.sigma = 7.
                self.paf_thre = 8.  # it is original 1.0 * stride in this program

        self.transform_params = TransformationParams()

    def create_augmenters(self):
        return

    @staticmethod
    def ltr_parts(parts_dict):
        # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
        leftParts = [parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"]]
        rightParts = [parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"]]
        return leftParts, rightParts


class COCOSourceHeadConfig(COCOSourceConfig):


    def __init__(self, hdf5_source):

        super().__init__(hdf5_source)

    def convert_mask(self, mask, global_config):

        mask = super().convert_mask(mask, global_config)

        # we added head layer here but haven't marked it yet. Lets wipe mask for whole layer.
        HeadCenter = global_config.parts_dict['HeadCenter']
        mask[:,:, HeadCenter] = 0.

        return mask


    def convert(self, meta, global_config):

        return super().convert(meta, global_config)

#        joints = np.array(meta['joints'])




class MPIISourceHeadConfig:

    def __init__(self, hdf5_source):

        self.hdf5_source = hdf5_source
        self.parts = ["HeadTop", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne",
             "Rank", "Lhip", "Lkne", "Lank"]

        self.num_parts = len(self.parts)

        # for COCO neck is calculated like mean of 2 shoulders.
        self.parts_dict = dict(zip(self.parts, range(self.num_parts)))


    def convert(self, meta, global_config):

        joints = np.array(meta['joints'])

        result = np.zeros((joints.shape[0], global_config.num_parts, 3), dtype=np.float)
        result[:,:,2]=2.  # 2 - abstent, 1 visible, 0 - invisible

        for p in self.parts:

            if p in global_config.parts_dict:
                coco_id = self.parts_dict[p]
                global_id = global_config.parts_dict[p]
                result[:, global_id, :] = joints[:, coco_id, :]
            else:
                assert p == "HeadTop"

        HeadCenterC = global_config.parts_dict['HeadCenter']
        neckC = self.parts_dict['neck']
        HeadTopC = self.parts_dict['HeadTop']

        # no head center, we calculate it as average of shoulders
        # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        both_parts_known = (joints[:, HeadTopC, 2]<2)  &  (joints[:, neckC, 2]<2)
        result[both_parts_known, HeadCenterC, 0:2] = (joints[both_parts_known, neckC, 0:2] +
                                                    joints[both_parts_known, HeadTopC, 0:2]) / 2
        result[both_parts_known, HeadCenterC, 2] = np.minimum(joints[both_parts_known, neckC, 2],
                                                                 joints[both_parts_known, HeadTopC, 2])

        meta['joints'] = result

        meta['scale_provided'] = [ x * 328 / 200 for x in meta['scale_provided'] ]

        return meta

    def convert_mask(self, mask, global_config):

        mask = np.repeat(mask[:, :, np.newaxis], global_config.num_layers, axis=2)


        Leye = global_config.parts_dict['Leye']
        Lear = global_config.parts_dict['Lear']
        Reye = global_config.parts_dict['Reye']
        Rear = global_config.parts_dict['Rear']

        mask[:, :, (Leye,Lear,Reye,Rear) ] = 0.

        return mask

    def source(self):

        return self.hdf5_source


Configs["HeadCount"] = HeadCounterConfig
