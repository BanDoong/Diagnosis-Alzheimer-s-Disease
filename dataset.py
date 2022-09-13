import os
import torch
import pandas as pd
import numpy as np
import nibabel as nib


# from trans import generate_pair


def get_subdir(path, subj):
    subj_dir = os.path.join(path, subj)
    ses = os.listdir(subj_dir)
    for s in ses:
        if 'ses' in s:
            ses_dir = os.path.join(subj_dir, s)
            return ses_dir, s


def which_tool(tool, path, subj, mask=True, resize=False, t1_linear=False):
    ses_dir, s = get_subdir(path, subj)
    group_name = 'classification'

    if t1_linear:
        if tool == 'MRI':
            path_all = os.path.join(ses_dir, 't1_linear')
            data_path = path_all + '/' + subj + '_' + s + '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w.nii.gz'
        elif tool == 'Tau':
            path_all = os.path.join(ses_dir, 'pet_linear', 'coreg')
            data_path = path_all + '/' + subj + '_Tau.nii.gz'
        elif tool == 'Amyloid':
            path_all = os.path.join(ses_dir, 'pet_linear', 'coreg')
            data_path = path_all + '/' + subj + '_Amyloid.nii.gz'
    else:
        if tool == 'MRI':
            path_all = os.path.join(ses_dir, 't1/spm/segmentation/normalized_space')
            if mask and not resize:
                data_path = path_all + '/' + subj + '_' + s + '_space-Ixi549Space_T1w_mask_norm.nii.gz'
            elif resize:
                data_path = path_all + '/' + subj + '_' + s + '_space-Ixi549Space_T1w_resized.nii.gz'
            else:
                data_path = path_all + '/' + subj + '_' + s + '_space-Ixi549Space_T1w_norm.nii.gz'

        elif tool == 'Tau':
            path_all = os.path.join(ses_dir, 'pet/preprocessing/group-' + str(group_name))
            if mask:
                data_path = path_all + '/' + subj + '_adni_tau_coreg_mask_norm.nii.gz'
            else:
                data_path = path_all + '/' + subj + '_adni_tau_coreg_norm.nii.gz'

        elif tool == 'Amyloid':
            path_all = os.path.join(ses_dir, 'pet/preprocessing/group-' + str(group_name))
            if mask:
                data_path = path_all + '/' + subj + '_adni_amyloid_coreg_mask_norm.nii.gz'
            else:
                data_path = path_all + '/' + subj + '_adni_amyloid_coreg_norm.nii.gz'

    return data_path


def _get_data(dataset, dir_label, ad, mci, num_label, fold, mci_convert=False):
    if dataset == 'train':
        label_path = os.path.join(dir_label, 'train_splits-5', f'split-{fold}')
        data_label = ['CN.tsv', 'AD.tsv', 'MCI.tsv', 'sMCI.tsv', 'pMCI.tsv']
    else:
        label_path = os.path.join(dir_label, 'validation_splits-5', f'split-{fold}')
        data_label = ['CN_baseline.tsv', 'AD_baseline.tsv', 'MCI_baseline.tsv', 'sMCI_baseline.tsv',
                      'pMCI_baseline.tsv']
    if mci_convert:
        label = pd.read_csv(os.path.join(label_path, data_label[3]), sep='\t')
        label = pd.concat([label, pd.read_csv(os.path.join(label_path, data_label[4]), sep='\t')],
                          ignore_index=True)
    else:
        label = pd.read_csv(os.path.join(label_path, data_label[0]), sep='\t')
        if ad:
            label = pd.concat([label, pd.read_csv(os.path.join(label_path, data_label[1]), sep='\t')],
                              ignore_index=True)
        if mci:
            label = pd.concat([label, pd.read_csv(os.path.join(label_path, data_label[2]), sep='\t')],
                              ignore_index=True)

    # if dataset == 'train':
    #     label = label.iloc[:, [1, 3]]
    # else:
    #     label = label.iloc[:, [2, 4]]
    label = label.iloc[:, [0, 2]]
    print(label)

    for idx in range(len(label['diagnosis'])):
        if num_label == 3:
            if label['diagnosis'][idx] == 'AD' or label['diagnosis'][idx] == 'Dementia':
                label['diagnosis'][idx] = 2
            elif label['diagnosis'][idx] == 'MCI':
                label['diagnosis'][idx] = 1
            else:
                label['diagnosis'][idx] = 0
        else:
            # ad/mci vs cn
            if ad and mci:
                if label['diagnosis'][idx] == 'AD' or label['diagnosis'][idx] == 'MCI':
                    label['diagnosis'][idx] = 1
                else:
                    label['diagnosis'][idx] = 0
            # ad vs cn
            elif ad and not mci:
                if label['diagnosis'][idx] == 'AD' or label['diagnosis'][idx] == 'Dementia':
                    label['diagnosis'][idx] = 1
                else:
                    label['diagnosis'][idx] = 0
            else:
                # cn vs mci or sMCI vs pMCI
                if label['diagnosis'][idx] == 'MCI' or label['diagnosis'][idx] == 'pMCI':
                    label['diagnosis'][idx] = 1
                else:
                    label['diagnosis'][idx] = 0
    return label


def roi_which_tool(tool, path, subj, side=None, data_type='linear'):
    ses_dir, s = get_subdir(path, subj)
    if tool == 'MRI':
        path_all = os.path.join(ses_dir, f'roi_based/t1_{data_type}')
        data_path = path_all + '/' + subj + '_mri_roi_based_' + side + '_50.nii.gz'
    elif tool == 'Tau':
        path_all = os.path.join(ses_dir, f'roi_based/pet_{data_type}/tau')
        data_path = path_all + '/' + subj + '_tau_roi_based_' + side + '_50.nii.gz'
    elif tool == 'Amyloid':
        path_all = os.path.join(ses_dir, f'roi_based/pet_{data_type}/amyloid')
        data_path = path_all + '/' + subj + '_amyloid_roi_based_' + side + '_50.nii.gz'
    else:
        print('You have to choose one of MRI, Tau, Amyloid')
    return data_path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, fold, args, transformation):
        super(Dataset, self).__init__()

        self.dataset = dataset
        self.fold = fold
        self.finetune = args.finetune
        self.modality = args.modality
        self.args = args
        self.trans_tau = args.trans_tau
        self.transformation = transformation
        self.img_mri = []
        self.img_tau = []
        self.img_amyloid = []
        self.img_fdg = []
        self.img_mri_left = []
        self.img_mri_right = []
        self.img_tau_left = []
        self.img_tau_right = []
        self.img_amyloid_left = []
        self.img_amyloid_right = []
        self.img_fdg_left = []
        self.img_fdg_right = []
        self.label = []
        self.label_left = []
        self.label_right = []

        self.roi = args.roi

        if args.t1_linear:
            data_type = 'linear'
        else:
            data_type = 'volume'

        self.subj_list = _get_data(self.dataset, args.dir_label, args.ad, args.mci, args.num_label, fold,
                                   args.mci_convert)
        print(
            f'Total Subjects Number for Finetunning : {args.finetune}({self.dataset})  :   {len(self.subj_list["participant_id"])}')
        # self.subj_list = self.subj_list['diagnosis'].sort_index(ascending=True)
        if 'ADNI1' in args.dir_data:
            ban_list = ['100_S_0747', '003_S_1257', '031_S_0830']
        else:
            # ban_list = ['009_S_6212', '012_S_6073', '016_S_6809', '027_S_5079', '027_S_6001', '027_S_6842',
            #             '027_S_5109',
            #             '003_S_1122', '009_S_6402', '027_S_6002', '027_S_6317', '041_S_6785', '067_S_4782',
            #             '070_S_6236',
            #             '098_S_6593', '128_S_2002', '128_S_2123', '153_S_6755', '027_S_6463', '057_S_6869',
            #             '070_S_6911',
            #             '029_S_6798', '035_S_4464', '041_S_4510', '099_S_6038', '126_S_0680', '129_S_6784',
            #             '023_S_6334',
            #             '094_S_6278', '114_S_6524']
            ban_list = ['020_S_6227']
        self.num_data = 0
        for subj in list(self.subj_list['participant_id']):
            subj_origin = subj
            # if 'caps' not in args.dir_data and 'ADNI1' not in args.dir_data:
            #     subj = subj[8:11] + '_S_' + subj[12:16]
            if subj in ban_list:
                pass
            else:
                if 'caps' in args.dir_data:
                    if args.roi:
                        self.mri_list_left = roi_which_tool(tool='MRI', path=args.dir_data, subj=subj, side='left',
                                                            data_type=data_type)
                        self.mri_list_right = roi_which_tool(tool='MRI', path=args.dir_data, subj=subj, side='right',
                                                             data_type=data_type)
                        self.tau_list_left = roi_which_tool(tool='Tau', path=args.dir_data, subj=subj, side='left',
                                                            data_type=data_type)
                        self.tau_list_right = roi_which_tool(tool='Tau', path=args.dir_data, subj=subj, side='right',
                                                             data_type=data_type)
                        self.amyloid_list_left = roi_which_tool(tool='Amyloid', path=args.dir_data, subj=subj,
                                                                side='left',
                                                                data_type=data_type)
                        self.amyloid_list_right = roi_which_tool(tool='Amyloid', path=args.dir_data, subj=subj,
                                                                 side='right', data_type=data_type)

                    else:
                        self.mri_list = which_tool('MRI', path=args.dir_data, subj=subj, mask=True, resize=args.resize,
                                                   t1_linear=args.t1_linear)
                        self.tau_list = which_tool('Tau', path=args.dir_data, subj=subj, mask=True, resize=args.resize,
                                                   t1_linear=args.t1_linear)
                        self.amyloid_list = which_tool('Amyloid', path=args.dir_data, subj=subj, mask=True,
                                                       resize=args.resize, t1_linear=args.t1_linear)

                elif 'free' in args.dir_data:
                    if args.resize and not args.resize64:
                        self.mri_list = os.path.join(args.dir_data, subj, f'{subj}_MRI_mask_norm_resized.nii.gz')
                        self.tau_list = os.path.join(args.dir_data, subj, f'{subj}_Tau_mask_coreg_norm_resized.nii.gz')
                        self.amyloid_list = os.path.join(args.dir_data, subj,
                                                         f'{subj}_Amyloid_mask_coreg_norm_resized.nii.gz')
                    elif args.resize64 and not args.resize:
                        self.mri_list = os.path.join(args.dir_data, subj, f'{subj}_MRI_mask_norm_resized_64.nii.gz')
                        self.tau_list = os.path.join(args.dir_data, subj,
                                                     f'{subj}_Tau_mask_coreg_norm_resized_64.nii.gz')
                        self.amyloid_list = os.path.join(args.dir_data, subj,
                                                         f'{subj}_Amyloid_mask_coreg_norm_resized_64.nii.gz')
                    else:
                        self.mri_list = os.path.join(args.dir_data, subj, f'{subj}_MRI_mask_norm.nii.gz')
                        self.tau_list = os.path.join(args.dir_data, subj, f'{subj}_Tau_mask_coreg_norm.nii.gz')
                        self.amyloid_list = os.path.join(args.dir_data, subj, f'{subj}_Amyloid_mask_coreg_norm.nii.gz')
                else:
                    # ANTS & FSL
                    if args.crop:
                        if args.resize:
                            #
                            # self.mri_list = os.path.join(args.dir_data, subj,
                            #                              f'{subj}_MRI_mask_norm_cropped_resized.nii.gz')
                            # self.tau_list = os.path.join(args.dir_data, subj,
                            #                              f'{subj}_Tau_mask_norm_cropped_resized.nii.gz')
                            # self.amyloid_list = os.path.join(args.dir_data, subj,
                            #                                  f'{subj}_Amyloid_mask_norm_cropped_resized.nii.gz')
                            self.mri_list = os.path.join(args.dir_data, subj,
                                                         f'{subj}_MRI_mask_norm_crop_resize_aug_0.nii.gz')
                            self.tau_list = os.path.join(args.dir_data, subj,
                                                         f'{subj}_Tau_mask_norm_crop_resize_aug_0.nii.gz')
                            self.amyloid_list = os.path.join(args.dir_data, subj,
                                                             f'{subj}_Amyloid_mask_norm_crop_resize_aug_0.nii.gz')
                            self.fdg_list = os.path.join(args.dir_data, subj,
                                                         f'{subj}_fdg_mask_norm_crop_resize_aug_0.nii.gz')
                            if 'mri' in args.modality:
                                self.img_mri.append(
                                    np.expand_dims(nib.load(self.mri_list).get_fdata().astype('float32'), 0))
                            if 'tau' in args.modality:
                                self.img_tau.append(
                                    np.expand_dims(nib.load(self.tau_list).get_fdata().astype('float32'), 0))
                            if 'amyloid' in args.modality:
                                self.img_amyloid.append(
                                    np.expand_dims(nib.load(self.amyloid_list).get_fdata().astype('float32'),
                                                   0))
                            if 'fdg' in args.modality:
                                self.img_amyloid.append(
                                    np.expand_dims(nib.load(self.fdg_list).get_fdata().astype('float32'),
                                                   0))
                            self.label += [
                                int(self.subj_list[self.subj_list['participant_id'] == subj_origin]['diagnosis'])]
                            self.num_data += 1
                        else:
                            self.mri_list = os.path.join(args.dir_data, subj, f'{subj}_MRI_mask_norm_crop.nii.gz')
                            self.tau_list = os.path.join(args.dir_data, subj, f'{subj}_Tau_mask_norm_crop.nii.gz')
                            self.amyloid_list = os.path.join(args.dir_data, subj,
                                                             f'{subj}_Amyloid_mask_norm_crop.nii.gz')
                            self.fdg_list = os.path.join(args.dir_data, subj,
                                                         f'{subj}_fdg_mask_norm_crop.nii.gz')
                            self.label += [
                                int(self.subj_list[self.subj_list['participant_id'] == subj_origin]['diagnosis'])]
                            self.num_data += 1
                    elif args.roi:
                        self.mri_list_left = os.path.join(args.dir_data, subj,
                                                          f'{subj}_MRI_{args.norm}_{args.region}_left.nii.gz')
                        self.tau_list_left = os.path.join(args.dir_data, subj,
                                                          f'{subj}_Tau_{args.norm}_{args.region}_left.nii.gz')
                        self.amyloid_list_left = os.path.join(args.dir_data, subj,
                                                              f'{subj}_Amyloid_{args.norm}_{args.region}_left.nii.gz')
                        self.fdg_list_left = os.path.join(args.dir_data, subj,
                                                          f'{subj}_fdg_{args.norm}_{args.region}_left.nii.gz')

                        self.mri_list_right = os.path.join(args.dir_data, subj,
                                                           f'{subj}_MRI_{args.norm}_{args.region}_right.nii.gz')
                        self.tau_list_right = os.path.join(args.dir_data, subj,
                                                           f'{subj}_Tau_{args.norm}_{args.region}_right.nii.gz')
                        self.amyloid_list_right = os.path.join(args.dir_data, subj,
                                                               f'{subj}_Amyloid_{args.norm}_{args.region}_right.nii.gz')
                        self.fdg_list_right = os.path.join(args.dir_data, subj,
                                                           f'{subj}_fdg_{args.norm}_{args.region}_right.nii.gz')

                        self.label += [
                            int(self.subj_list[self.subj_list['participant_id'] == subj_origin]['diagnosis'])]
                        self.num_data += 1
                    else:
                        self.mri_list = os.path.join(args.dir_data, subj, f'{subj}_MRI_mask_norm.nii.gz')
                        self.tau_list = os.path.join(args.dir_data, subj, f'{subj}_Tau_mask_norm.nii.gz')
                        self.amyloid_list = os.path.join(args.dir_data, subj, f'{subj}_Amyloid_mask_norm.nii.gz')
                        self.fdg_list = os.path.join(args.dir_data, subj, f'{subj}_fdg_mask_norm.nii.gz')

                if 'mri' in args.modality:
                    if args.roi:
                        self.img_mri_left.append(
                            np.expand_dims(nib.load(self.mri_list_left).get_fdata().astype('float32'), 0))
                        self.img_mri_right.append(
                            np.expand_dims(nib.load(self.mri_list_right).get_fdata().astype('float32'), 0))
                    else:
                        self.img_mri.append(np.expand_dims(nib.load(self.mri_list).get_fdata().astype('float32'), 0))
                if 'tau' in args.modality:
                    if args.roi:
                        self.img_tau_left.append(
                            np.expand_dims(nib.load(self.tau_list_left).get_fdata().astype('float32'), 0))
                        self.img_tau_right.append(
                            np.expand_dims(nib.load(self.tau_list_right).get_fdata().astype('float32'), 0))
                    else:
                        self.img_tau.append(np.expand_dims(nib.load(self.tau_list).get_fdata().astype('float32'), 0))
                if 'amyloid' in args.modality:
                    if args.roi:
                        self.img_amyloid_left.append(
                            np.expand_dims(nib.load(self.amyloid_list_left).get_fdata().astype('float32'), 0))
                        self.img_amyloid_right.append(
                            np.expand_dims(nib.load(self.amyloid_list_right).get_fdata().astype('float32'), 0))
                    else:
                        self.img_amyloid.append(
                            np.expand_dims(nib.load(self.amyloid_list).get_fdata().astype('float32'), 0))
                if 'fdg' in args.modality:
                    if args.roi:
                        self.img_fdg_left.append(
                            np.expand_dims(nib.load(self.fdg_list_left).get_fdata().astype('float32'), 0))
                        self.img_fdg_right.append(
                            np.expand_dims(nib.load(self.fdg_list_right).get_fdata().astype('float32'), 0))
                    else:
                        self.img_fdg.append(
                            np.expand_dims(nib.load(self.fdg_list).get_fdata().astype('float32'), 0))
        if self.roi:
            # self.label_left = list(self.subj_list['diagnosis'])
            # self.label_right = list(self.subj_list['diagnosis'])
            if len(self.label_left) == len(self.label_right) == len(self.subj_list['participant_id']):
                print(f"Label and Subjects numbers are same as {len(self.subj_list['participant_id'])}")
        else:
            # self.label = list(self.subj_list['diagnosis'])
            if len(self.label) == len(self.subj_list['participant_id']):
                print(f"Label and Subjects numbers are same as {len(self.subj_list['participant_id'])}")

    def __getitem__(self, index):
        # print(index)
        # print(type(index))
        img_mri_i, img_mri_j, img_tau_i, img_tau_j, img_amyloid_i, img_amyloid_j = 1, 1, 1, 1, 1, 1
        if self.roi:
            img_mri_left = self.img_mri_left[index] if 'mri' in self.modality else 1
            img_mri_right = self.img_mri_right[index] if 'mri' in self.modality else 1
            img_tau_left = self.img_tau_left[index] if 'tau' in self.modality else 1
            img_tau_right = self.img_tau_right[index] if 'tau' in self.modality else 1
            img_amyloid_left = self.img_amyloid_left[index] if 'amyloid' in self.modality else 1
            img_amyloid_right = self.img_amyloid_right[index] if 'amyloid' in self.modality else 1
            img_fdg_left = self.img_fdg_left[index] if 'fdg' in self.modality else 1
            img_fdg_right = self.img_fdg_right[index] if 'fdg' in self.modality else 1
        else:
            img_mri = self.img_mri[index] if 'mri' in self.modality else 1
            img_tau = self.img_tau[index] if 'tau' in self.modality else 1
            img_amyloid = self.img_amyloid[index] if 'amyloid' in self.modality else 1
            img_fdg = self.img_fdg[index] if 'fdg' in self.modality else 1
        label = self.label[index]

        if self.roi:
            sample = {'img_mri_left': img_mri_left, 'img_mri_right': img_mri_right, 'img_tau_left': img_tau_left,
                      'img_tau_right': img_tau_right, 'img_amyloid_left': img_amyloid_left,
                      'img_amyloid_right': img_amyloid_right, 'img_fdg_left': img_fdg_left,
                      'img_fdg_right': img_fdg_right, 'label': label}

        else:
            sample = {'img_mri': img_mri, 'img_tau': img_tau, 'img_amyloid': img_amyloid, 'img_fdg': img_fdg,
                      'label': label}

        return sample

    def __len__(self):
        # if self.roi:
        #     return len(self.subj_list) * 2
        # else:
        # return len(self.subj_list)
        return self.num_data


class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())
