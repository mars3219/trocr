import os
import cv2
import numpy as np
import pandas as pd
import glob
import time
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt

import argparse

# 1. 이미지 파일 리스트업
def list_img_files(folder_path):
    return glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True)


def compare_images_ssim(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = ssim(gray_image1, gray_image2, full=True)
    diff = (diff * 255).astype("uint8")
    
    return score, diff


# def opencv_ssim(image1, image2):
#     hists = []
#     for img in [image1, image2]:
#         # BGR 이미지를 HSV 이미지로 변환
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#         # 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
#         hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
#         # 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
#         cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
#         # hists 리스트에 저장
#         hists.append(hist)

#     # 1번째 이미지를 원본으로 지정
#     query = hists[0]

#     # 비교 알고리즘의 이름들을 리스트에 저장
#     methods = ['CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA', 'EMD']

#     # 5회 반복(5개 비교 알고리즘을 모두 사용)
#     for index, name in enumerate(methods):
#         # 비교 알고리즘 이름 출력(문자열 포맷팅 및 탭 적용)
#         print('%-10s' % name, end = '\t')  
        
#         # 2회 반복(2장의 이미지에 대해 비교 연산 적용)
#         for i, histogram in enumerate(hists):
#             ret = cv2.compareHist(query, histogram, index) 
            
#             if index == cv2.HISTCMP_INTERSECT:                   # 교차 분석인 경우 
#                 ret = ret/np.sum(query)                          # 원본으로 나누어 1로 정규화
                
#             print("img%d :%7.2f"% (i+1 , ret), end='\t')        # 비교 결과 출력


def main(folder_path, output_csv, save_path):
    image_files = list_img_files(folder_path)
    # image_arrays = {img: image_to_array(load_image(os.path.join(folder_path, img))) for img in image_files}

    results = []
    cnt = 0
    temp = 0

    for i, img1 in enumerate(image_files):
        cnt += 1
        start_time = time.time()
        print("===========================================================================")
        print(f"{cnt}/{len(image_files)-1}번째 이미지: 유사도 비교 시작")

        res = []
        for j, img2 in enumerate(image_files):
            if i < j:  # 중복 계산 방지
                image1 = cv2.imread(img1)
                image2 = cv2.imread(img2)
                
                if image1.shape != image2.shape:
                    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

                # Compare images using SSIM
                ssim_score, diff_image = compare_images_ssim(image1, image2)
                # ssim_score, diff_image = opencv_ssim(image1, image2)

                res.append((img1, img2, ssim_score))
                results.append((img1, img2, ssim_score))
                temp += 1
                print(f"{temp} / {len(image_files)-1}   유사도 점수: {ssim_score: .2f}")


                plt.subplot(1, 3, 1)
                plt.title('Image 1')
                plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

                plt.subplot(1, 3, 2)
                plt.title('Image 2')
                plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

                plt.subplot(1, 3, 3)
                plt.title('Difference')
                plt.imshow(diff_image, cmap='gray')

                # plt.show()
                
                plt.tight_layout()

                if save_path is not None:
                    img_file = os.path.join(save_path, f"ssim_img{temp}.jpg")
                    plt.savefig(img_file)
                else:
                    plt.show()
                plt.close()

                img = cv2.imread(img_file)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (0, 255, 0)
                thickness = 1 

                # cv2.putText(img, f"origin: {img1}", (50,50), font, font_scale, font_color, thickness)
                # cv2.putText(img, f"compare: {img2}", (50,100), font, font_scale, font_color, thickness)
                cv2.putText(img, f"similatity: {ssim_score: .2f}", (50,150), font, font_scale, (255,0,0), thickness)
                cv2.imwrite(img_file, img)


        end_time = time.time()
        duration = end_time - start_time
        print(f"{cnt}/{len(image_files)-1}번째 이미지: 유사도 비교 완료")
        print(f"처리시간: {duration: .4f}")
        print("===========================================================================")

        # 4. 결과 저장
        temp_csv = os.path.join(output_csv, f"{img1}.csv")
        df = pd.DataFrame(res, columns=['Image1', 'Image2', 'ssim_score'])
        df.to_csv(temp_csv, index=False)
        res.clear()


    # 4. 결과 저장
    output = os.path.join(output_csv, f"total_ssim.csv")
    df = pd.DataFrame(results, columns=['Image1', 'Image2', 'ssim_score'])
    df.to_csv(output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate image similarities and save results to CSV')
    parser.add_argument('--img_path', type=str, default='/workspace/unilm/trocr/당진', help='Path to the folder contain')
    parser.add_argument('--output_csv', type=str, default='/workspace/unilm/trocr/output_csv', help='Path to the output CSV file')
    parser.add_argument('--save_img_path', type=str, default='/workspace/unilm/trocr/output_img', help='Path to the output img file')
    

    args = parser.parse_args()

    main(args.img_path, args.output_csv, args.save_img_path)

