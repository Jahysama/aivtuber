import sys
sys.path.append("MakeItTalk/thirdparty/AdaptiveWingLoss")
import os, glob
import numpy as np
import cv2
import argparse
import pickle

from thirdparty.resemblyer_util.speaker_emb import get_spk_emb


default_head_name = 'paint_boy'  # the image name (with no .jpg) to animate
ADD_NAIVE_EYE = True  # whether add naive eye blink
CLOSE_INPUT_FACE_MOUTH = False  # if your image has an opened mouth, put this as True, else False
AMP_LIP_SHAPE_X = 2.  # amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_Y = 2.  # amplify the lip motion in vertical direction
AMP_HEAD_POSE_MOTION = 0.7  # amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='{}.jpg'.format(default_head_name))
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str,
                    default='examples/ckpt/ckpt_content_branch.pth')  # ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str,
                    default='examples/ckpt/ckpt_116_i2i_comb.pth')  # ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+',
                    default=[])  # ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')

parser.add_argument('--test_end2end', default=False, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')
parser.add_argument('-f')

opt_parser = parser.parse_args()


def get_talking_head(audio, face_landmarks, c, model):
    shape_3d, scale, shift = face_landmarks

    au_data = []
    au_emb = []
    ains = glob.glob1('examples', '*.wav')
    ains = [item for item in ains if item is not 'tmp.wav']
    ains.sort()


    img = cv2.imread('examples/' + opt_parser.jpg)

    for ain in ains:
        me, ae = get_spk_emb(audio)
        au_emb.append(me.reshape(-1))

        au_data_i = c.convert_single_wav_to_autovc_input(audio_filename=os.path.join('examples', ain),
                                                         autovc_model_path=opt_parser.load_AUTOVC_name)
        au_data += au_data_i

    # landmark fake placeholder
    fl_data = []
    rot_tran, rot_quat, anchor_t_shape = [], [], []
    for au, info in au_data:
        au_length = au.shape[0]
        fl = np.zeros(shape=(au_length, 68 * 3))
        fl_data.append((fl, info))
        rot_tran.append(np.zeros(shape=(au_length, 3, 4)))
        rot_quat.append(np.zeros(shape=(au_length, 4)))
        anchor_t_shape.append(np.zeros(shape=(au_length, 68 * 3)))

    if (os.path.exists(os.path.join( 'examples', 'dump', 'random_val_fl.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_fl.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_fl_interp.pickle'))):
        os.remove(os.path.join( 'examples', 'dump', 'random_val_fl_interp.pickle'))
    if (os.path.exists(os.path.join('examples', 'dump', 'random_val_au.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_au.pickle'))
    if (os.path.exists(os.path.join('MakeItTalk', 'examples', 'dump', 'random_val_gaze.pickle'))):
        os.remove(os.path.join('examples', 'dump', 'random_val_gaze.pickle'))

    with open(os.path.join('examples', 'dump', 'random_val_fl.pickle'), 'wb') as fp:
        pickle.dump(fl_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_au.pickle'), 'wb') as fp:
        pickle.dump(au_data, fp)
    with open(os.path.join('examples', 'dump', 'random_val_gaze.pickle'), 'wb') as fp:
        gaze = {'rot_trans': rot_tran, 'rot_quat': rot_quat, 'anchor_t_shape': anchor_t_shape}
        pickle.dump(gaze, fp)

    if(len(opt_parser.reuse_train_emb_list) == 0):
        model.test(au_emb=au_emb)
    else:
        model.test(au_emb=None)

    fls = glob.glob1('examples', 'pred_fls_*.txt')
    fls.sort()

    # for i in range(0,len(fls)):
    #     fl = np.loadtxt(os.path.join('examples', fls[i])).reshape((-1, 68,3))
    #     fl[:, :, 0:2] = -fl[:, :, 0:2]
    #     fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift
    #
    #     if (ADD_NAIVE_EYE):
    #         fl = util.add_naive_eye(fl)
    #
    #     # additional smooth
    #     fl = fl.reshape((-1, 204))
    #     fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
    #     fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
    #     fl = fl.reshape((-1, 68, 3))
    #
    #     ''' STEP 6: Imag2image translation '''
    #     model = Image_translation_block(opt_parser, single_test=True)
    #     with torch.no_grad():
    #         video = model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=opt_parser.jpg.split('.')[0])
    #         print('finish image2image gen')
    #     os.remove(os.path.join('examples', fls[i]))

    return fls

