# modified from ArSSR github repo

import os
import model
import torch
import argparse
import data
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt


def save_and_visualize_patch(lr_img, hr_img, pred_img, ssim_img, filename, visualize_SSIM=False):
    mid_slice = lr_img.shape[0] // 2  
    lr_slice = lr_img[mid_slice, :, :]
    hr_slice = hr_img[mid_slice, :, :]
    pred_slice = pred_img[mid_slice, :, :]
    ssim_slice = ssim_img[mid_slice, :, :]

    if visualize_SSIM:
        plt.figure(figsize=(32, 8))
    else:
        plt.figure(figsize=(24, 8))
    
    plt.subplot(1, 4, 1)
    plt.imshow(lr_slice, cmap='gray')
    plt.title('Low Resolution')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(hr_slice, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(pred_slice, cmap='gray')
    plt.title('Predicted')
    plt.axis('off')
    
    if visualize_SSIM:
        plt.subplot(1, 4, 4)
        plt.imshow(ssim_slice, cmap='gray')
        plt.title('SSIM Map')
        plt.axis('off')
    
    plt.savefig(filename)
    plt.close()

def adjust_state_dict_keys(state_dict, old_prefix, new_prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(old_prefix):
            new_key = key.replace(old_prefix, new_prefix, 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-depth', type=int, default=8, dest='decoder_depth',
                        help='the depth of the decoder network (default=8).')
    parser.add_argument('-width', type=int, default=256, dest='decoder_width',
                        help='the width of the decoder network (default=256).')
    parser.add_argument('-feature_dim', type=int, default=128, dest='feature_dim',
                        help='the dimension size of the feature vector (default=128)')
    parser.add_argument('-pre_trained_model', type=str, default='./model3/model_param_50.pkl',
                        dest='pre_trained_model', help='the file path of the trained model')
    parser.add_argument('-is_gpu', type=int, default=1, dest='is_gpu',
                        help='enable GPU (1->enable, 0->disable)')
    parser.add_argument('-gpu', type=int, default=0, dest='gpu',
                        help='the number of GPU')
    parser.add_argument('-input_path', type=str, default='./data/test', dest='input_path',
                        help='the file path of LR input image')
    parser.add_argument('-output_path', type=str, default='./data/nerfoutput', dest='output_path',
                        help='the file save path of reconstructed result')
    parser.add_argument('-scale', type=float, default='2.0', dest='scale',
                        help='the up-sampling scale k')
    parser.add_argument('-hr_data_test', type=str, default='./data/test', dest='hr_data_test',
                        help='the file path of HR patches for testing')
    parser.add_argument('-batch_size', type=int, default=1, dest='batch_size',
                        help='the batch size for testing')

    args = parser.parse_args()
    encoder_name = args.encoder_name
    decoder_depth = args.decoder_depth
    decoder_width = args.decoder_width
    feature_dim = args.feature_dim
    pre_trained_model = args.pre_trained_model
    gpu = args.gpu
    is_gpu = args.is_gpu
    input_path = args.input_path
    output_path = args.output_path
    scale = args.scale
    hr_data_test = args.hr_data_test
    batch_size = args.batch_size

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Create output directory for patches if it doesn't exist
    patches_output_path = os.path.join(output_path, 'patches')
    if not os.path.exists(patches_output_path):
        os.makedirs(patches_output_path)

    if is_gpu == 1 and torch.cuda.is_available():
        DEVICE = torch.device('cuda:{}'.format(str(gpu)))
    else:
        DEVICE = torch.device('cpu')
    mrirecon_model = model.mrirecon(feature_dim=feature_dim,
                           decoder_depth=int(decoder_depth / 2),
                           decoder_width=decoder_width).to(DEVICE)

    state_dict = torch.load(pre_trained_model, map_location=DEVICE)
    new_state_dict = adjust_state_dict_keys(state_dict, 'stage_one', 'model')
    new_state_dict = adjust_state_dict_keys(new_state_dict, 'stage_two', 'model')

    mrirecon_model.load_state_dict(new_state_dict)


    test_loader = data.loader_train(in_path_hr=hr_data_test, batch_size=batch_size,
                                    sample_size=1000, is_train=False)

    mrirecon_model.eval()
    total_ssim = 0
    total_psnr = 0
    total_mse = 0
    count = 0
    with torch.no_grad():
        for i, (img_lr, xyz_hr, img_hr) in enumerate(test_loader):
            img_lr = img_lr.unsqueeze(1).float().to(DEVICE)  # N×1×h×w×d
            xyz_hr = xyz_hr.view(1, -1, 3).float().to(DEVICE)  # N×K×3 (K=H×W×D)
            img_hr = img_hr.squeeze(1).cpu().numpy()  # Remove the added channel dimension

            img_pre = mrirecon_model(img_lr, xyz_hr).cpu().detach().numpy().reshape(img_hr.shape)
            
            for i in range(img_hr.shape[0]):
                ssim_val, ssim_img = ssim(img_hr[i], img_pre[i], data_range=img_pre[i].max() - img_pre[i].min(), full=True)
                psnr_val = psnr(img_hr[i], img_pre[i], data_range=img_pre[i].max() - img_pre[i].min())
                mse_val = mse(img_hr[i], img_pre[i])
                total_ssim += ssim_val
                total_psnr += psnr_val
                total_mse += mse_val
                count += 1
                try:
                    patch_filename = os.path.join(patches_output_path, f'{count}_{i}_comparison.png')
                    save_and_visualize_patch(img_lr.squeeze().cpu().numpy(), img_hr[i], img_pre[i], ssim_img, patch_filename)
                except Exception as e:
                    print(f'Error saving file : {e}')


    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    avg_mse = total_mse / count
    print(f'Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.2f}, Average MSE: {avg_mse:.4f}')
