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

METHODS= ['CORREL', 'CHISQR', 'INTERSECT', 'BHATTACHARYYA', 'EMD']
SSIM_THRES = 0.6
INTERSECT_THRES = 0.8
BHA_THRES = 0.7


def list_img_files(folder_path):
    return glob.glob(os.path.join(folder_path, '**', '*.png'), recursive=True)


def concatenate_image_with_text(images, save_path, ssim_results, flag=True):
    if flag:
        result = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
        
        current_x = 150
        for i, image in enumerate(images):
            height, width = image.shape[:2]
            height, width = int(height/4), int(width/4)
            image = cv2.resize(image, (width, height))
            
            if i == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
            result[450:450+height, current_x:current_x+width] = image
            current_x += width + 50
        
        # 텍스트 추가
        p = 50
        cv2.putText(result, f"ssim_score: {ssim_results[2]: .2f}", (50, p), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        for i, t in enumerate(ssim_results[5:7]):
            p += 50
            cv2.putText(result, f"{METHODS[i+2]}: {t: .2f}", (50, p), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        if ssim_results[2] > SSIM_THRES and ssim_results[5] > INTERSECT_THRES and (1-ssim_results[6]) > BHA_THRES:
            cv2.putText(result, f"Similaraty: YES", (50, p+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ssim = 1
        else:
            cv2.putText(result, f"Similaraty: NO", (50, p+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            ssim = 0
        
        # 결과 이미지를 파일로 저장
        file_name = ""
        for e, i in enumerate(ssim_results[:2]):
            if e == 1: file_name += "_vs_"
            file_name_with_ext = os.path.basename(i)
            file_name += os.path.splitext(file_name_with_ext)[0]
        file_path = os.path.join(save_path, f"{file_name}.jpg")
        cv2.imwrite(file_path, result)
        return ssim
    else:
        if ssim_results[2] > SSIM_THRES and ssim_results[5] > INTERSECT_THRES and (1-ssim_results[6]) > BHA_THRES:
            return 1
        else:
            return 0



def compare_images_ssim(image1, image2):
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = ssim(gray_image1, gray_image2, full=True)
    diff = (diff * 255).astype("uint8")
    
    return score, diff


def opencv_ssim(image1, image2):
    hists = []
    for img in [image1, image2]:
        # BGR 이미지를 HSV 이미지로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 히스토그램 연산(파라미터 순서 : 이미지, 채널, Mask, 크기, 범위)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # 정규화(파라미터 순서 : 정규화 전 데이터, 정규화 후 데이터, 시작 범위, 끝 범위, 정규화 알고리즘)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        # hists 리스트에 저장
        hists.append(hist)

    # 1번째 이미지를 원본으로 지정
    query = hists[0]

    result = []
    # 5회 반복(5개 비교 알고리즘을 모두 사용)
    for index, name in enumerate(METHODS):
        # 비교 알고리즘 이름 출력(문자열 포맷팅 및 탭 적용)
        # print('%-10s' % name, end = '\t')  
        
        # 2회 반복(2장의 이미지에 대해 비교 연산 적용)
        for i, histogram in enumerate(hists):
            ret = cv2.compareHist(query, histogram, index) 
            
            if index == cv2.HISTCMP_INTERSECT:                   # 교차 분석인 경우 
                ret = ret/np.sum(query)                          # 원본으로 나누어 1로 정규화
                
            if i == 1:
                # print("img%d :%7.2f"% (i+1 , ret), end='\t')        # 비교 결과 출력
                result.append((name, ret))
        # print('')
    return result


def main(folder_path, output_csv, save_path):
    image_files = list_img_files(folder_path)
    results = []
    cnt = 0

    for i, img1 in enumerate(image_files):
        cnt += 1
        temp = 0

        if len(image_files) == cnt: # 마지막 파일일 경우 비교할 대상이 없음
            # 전체 결과 저장
            output = os.path.join(output_csv, f"total_ssim.csv")
            df = pd.DataFrame(results, columns=['origin_path', 'compare_path', 'ssim_score', 'correl_score', 'chisqr_score', 'intersect', 'bhattacharyya_score', 'emd_score', 'similataty_yn'])
            df.to_csv(output, index=False)
            return 0

        start_time = time.time()
        print("===========================================================================")
        print(f"{cnt}/{len(image_files)-1}번째 이미지: 유사도 비교 시작")

        res = []
        for j, img2 in enumerate(image_files):
            if i < j:  # 중복 계산 방지
                image1 = cv2.imread(img1)
                image2 = cv2.imread(img2)
                
                image1 = cv2.resize(image1, (1920, 1080))
                image2 = cv2.resize(image2, (1920, 1080))

                # Compare images using SSIM
                ssim_score, diff_image = compare_images_ssim(image1, image2)
                opencv_score = opencv_ssim(image1, image2)

                ssim_results = (img1, img2, ssim_score, opencv_score[0][1], opencv_score[1][1], opencv_score[2][1], opencv_score[3][1], opencv_score[4][1])
                temp += 1
                print(f"{temp} / {len(image_files)-1} ssim 점수:\t\t{ssim_score: .2f}")
                print(f"{temp} / {len(image_files)-1} correl 점수:\t\t{opencv_score[0][1]: .2f}")
                print(f"{temp} / {len(image_files)-1} chisqr 점수:\t\t{opencv_score[1][1]: .2f}")
                print(f"{temp} / {len(image_files)-1} intersect 점수:\t{opencv_score[2][1]: .2f}")
                print(f"{temp} / {len(image_files)-1} bhattacharyya 점수:\t{1-opencv_score[3][1]: .2f}")
                print(f"{temp} / {len(image_files)-1} emd 점수:\t\t{opencv_score[4][1]: .2f}")
                print('')


                images = [image1, image2, diff_image]
                result = concatenate_image_with_text(images, save_path, ssim_results, flag=False)   # 이미지 생성(flag): True
                res_l = list(ssim_results)
                res_l.append(result)
                res_l = tuple(res_l)
                res.append(res_l)
                results.append(res_l)

        end_time = time.time()
        duration = end_time - start_time
        print(f"{cnt}/{len(image_files)-1}번째 이미지: 유사도 비교 완료")
        print(f"처리시간: {duration: .4f}")
        print("===========================================================================")

        # # 이미지 장당 결과 저장
        # file_name_with_ext = os.path.basename(img1)
        # file_name = os.path.splitext(file_name_with_ext)[0]
        # temp_csv = os.path.join(output_csv, f"{file_name}.csv")
        # df = pd.DataFrame(res, columns=['origin_path', 'compare_path', 'ssim_score', 'correl_score', 'chisqr_score', 'intersect', 'bhattacharyya_score', 'emd_score', 'similataty_yn'])
        # df.to_csv(temp_csv, index=False)

        res.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate image similarities and save results to CSV')
    parser.add_argument('--img_path', type=str, default='/workspace/unilm/trocr/당진', help='Path to the folder contain')
    parser.add_argument('--output_csv', type=str, default='/workspace/unilm/trocr/output_csv', help='Path to the output CSV file')
    parser.add_argument('--save_img_path', type=str, default='/workspace/unilm/trocr/output_img', help='Path to the output img file')
    

    args = parser.parse_args()

    main(args.img_path, args.output_csv, args.save_img_path)

