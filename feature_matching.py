import cv2
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
img1 = cv2.imread('/workspace/unilm/trocr/당진 copy/CSU B.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/workspace/unilm/trocr/당진 copy/CSU C.png', cv2.IMREAD_GRAYSCALE)

# 2. SIFT 검출기 생성
sift = cv2.SIFT_create()

# 3. 특징점과 디스크립터 계산
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 4. BFMatcher 생성 및 매칭
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 5. 매칭 결과를 거리 순으로 정렬
matches = sorted(matches, key = lambda x:x.distance)

# 매칭 결과를 그리기
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img_matches)
plt.title('Feature Matching')
plt.show()

# 6. 유사도 평가 (여기서는 매칭된 특징점 수를 유사도로 사용)
similarity = len(matches)
print(f'유사도 (매칭된 특징점 수): {similarity}')
