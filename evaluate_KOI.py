import os
import argparse
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from tqdm import tqdm

# LPIPS 모델 로드 (AlexNet을 사용)
loss_fn = lpips.LPIPS(net='alex')

# 이미지 비교 함수
def calculate_metrics(img1, img2):
    # PSNR 계산
    psnr_value = psnr(img1, img2)
    
    # SSIM 계산 (멀티채널 이미지를 고려하여 channel_axis=2 설정)
    ssim_value = ssim(img1, img2, channel_axis=2)
    
    # LPIPS 계산 (이미지를 텐서로 변환)
    img1_tensor = lpips.im2tensor(img1)
    img2_tensor = lpips.im2tensor(img2)
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

# 이미지 폴더 비교 함수
def compare_images_in_folder(render_dir, gt_dir):
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # 폴더 내의 이미지 파일 개수 확인
    render_files = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    num_images = min(len(render_files), len(gt_files))

    for img_name in tqdm(render_files[:num_images]):  # 최대 num_images까지 비교
        render_img_path = os.path.join(render_dir, img_name)
        gt_img_path = os.path.join(gt_dir, img_name)

        # 이미지 로드
        render_img = Image.open(render_img_path).convert('RGB')
        gt_img = Image.open(gt_img_path).convert('RGB')

        # 이미지를 numpy 배열로 변환
        render_img_np = np.array(render_img)
        gt_img_np = np.array(gt_img)

        # 메트릭 계산
        psnr_value, ssim_value, lpips_value = calculate_metrics(render_img_np, gt_img_np)

        # 결과 저장
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)

        # 진행 상황 출력
        print(f"Processed {img_name}: PSNR={psnr_value:.4f}, SSIM={ssim_value:.4f}, LPIPS={lpips_value:.4f}")

    # 평균 결과 반환
    return np.mean(psnr_values), np.mean(ssim_values), np.mean(lpips_values)

# 메인 실행 함수
def main(m):
    # test/ours_10000 폴더 비교
    test_render_dir = os.path.join(m, 'test', 'ours_10000', 'renders')
    test_gt_dir = os.path.join(m, 'test', 'ours_10000', 'gt')
    print("\nComparing images in test/ours_10000:")
    test_psnr, test_ssim, test_lpips = compare_images_in_folder(test_render_dir, test_gt_dir)

    # train/ours_10000 폴더 비교
    train_render_dir = os.path.join(m, 'train', 'ours_10000', 'renders')
    train_gt_dir = os.path.join(m, 'train', 'ours_10000', 'gt')
    print("\nComparing images in train/ours_10000:")
    train_psnr, train_ssim, train_lpips = compare_images_in_folder(train_render_dir, train_gt_dir)

    # 최종 평균 결과 출력
    print("\nFinal Results:")
    print(f"Test Set - Average PSNR: {test_psnr:.4f}, Average SSIM: {test_ssim:.4f}, Average LPIPS: {test_lpips:.4f}")
    print(f"Train Set - Average PSNR: {train_psnr:.4f}, Average SSIM: {train_ssim:.4f}, Average LPIPS: {train_lpips:.4f}")

# 실행 인자로 경로를 받아서 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image quality metric calculation')
    parser.add_argument('m', type=str, help='Base path for test and train folders')
    args = parser.parse_args()

    main(args.m)
