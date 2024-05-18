import cv2
import numpy as np
import glob

def calculate_histogram(image, bins=256):
    """计算图像的颜色直方图"""
    histogram = cv2.calcHist([image], [0], None, [bins], [0, 256])
    return cv2.normalize(histogram, histogram).flatten()

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CHISQR):
    """比较两个直方图"""
    hist1 = hist1.astype('float32')
    hist2 = hist2.astype('float32')
    return cv2.compareHist(hist1, hist2, method)

def get_motion_mask(fg_mask, min_thresh=0, kernel=np.array((9,9), dtype=np.uint8)):
    _, thresh = cv2.threshold(fg_mask,min_thresh,255,cv2.THRESH_BINARY)
    if thresh is not None and thresh.size > 0:
        motion_mask = cv2.medianBlur(thresh, 3)
    else:
        motion_mask=fg_mask
        return motion_mask
    # morphological operations
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return motion_mask

def calculate(fg_mask,image):
    motion_mask = get_motion_mask(fg_mask, kernel=np.array((9,9), dtype=np.uint8))
    color_foreground = cv2.bitwise_and(image, image, mask=motion_mask)
    color_foreground=cv2.cvtColor(color_foreground, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始化一个累加直方图
    cumulative_histogram = np.zeros(256)
    hists=[]
    # 假设 contours 是你通过 findContours 得到的轮廓列表
    # 假设 image 是原始图像
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cropped_region = image[y:y+h, x:x+w]     
        # 计算裁剪区域的颜色直方图
        histogram = calculate_histogram(cropped_region)
        hists.append(histogram)
        # 累加到综合直方图中
        cumulative_histogram += histogram
    return cumulative_histogram,hists
def main():
    # 输入你的视频和查询图片文件夹的路径
    folder_path='' 
    video = ''
    standard_threshold = 40
    # get background subtractor
    sub_type = 'MOG2' # 'MOG2'
    if sub_type == "MOG2":
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True,history=800)
        # backSub.setShadowThreshold(0.75)
    else:
        backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=800, detectShadows=True)    
    query_inputs = glob.glob(folder_path + "/*")
    query_num=len(query_inputs)
    frame_count=0
    cap = cv2.VideoCapture(video)
    frame_count=0
    hists=[]
    count=[]
    count = [0] * query_num
    for i in range(query_num):
        hists.append(calculate_histogram(cv2.imread(query_inputs[i])))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while(1):
        ret, image = cap.read()
        if not ret:
            break
        frame_count=frame_count+1
        fg_mask = backSub.apply(image)
        # 将掩码应用于原始图像以获得彩色前景
        try:
            cumulative_histogram,histss=calculate(fg_mask,image)
        except Exception as e:
            print(frame_count)
            continue
        for i in range(query_num):
            # 比较直方图
            similarity = compare_histograms(cumulative_histogram, hists[i])
            if similarity<standard_threshold:
                count[i]=count[i]+1
    # 打印每个查询过滤后剩余的帧数
    print(count)


if __name__ == "__main__":
    main()
