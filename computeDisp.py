import numpy as np
import cv2.ximgproc as xip
import cv2
import time

def adaptive_median_filter(img, max_window_size):
    # 獲取圖像的高度和寬度
    height, width = img.shape
    # 創建一個與原圖像大小相同的空白圖像
    output = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            # 初始window大小為3
            window_size = 3
            # 不斷增加window大小直到達到最大window大小
            while window_size <= max_window_size:
                # 計算window半徑
                window_radius = window_size // 2
                # 根據window大小取得相應區域的像素值
                window = img[max(i-window_radius,0):min(i+window_radius+1,height),
                             max(j-window_radius,0):min(j+window_radius+1,width)]
                # 在window中找到最小值、最大值以及對應的索引
                min_val, max_val, min_idx, max_idx = cv2.minMaxLoc(window)
                # 如果當前像素值在window的最小值和最大值之間，則跳出循環
                if min_val < img[i,j] < max_val:
                    break
                # 增加window大小
                window_size += 1

            # 如果window大小超過最大window大小，將輸出圖像的像素值設置為當前像素值減去10，如果小於0則設置為0
            if window_size > max_window_size:
                output[i,j] = max(img[i,j] - 10, 0)
            else:
                # 否則，將輸出圖像的像素值設置為窗口中位數的值
                median_val = np.median(window)
                output[i,j] = median_val
    # 返回適應性中值濾波後的圖像
    return output

def computeDisp(Il, Ir, max_disp):
    start_time = time.time()
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    
    # Padding first
    border_size = 1
    border_type = cv2.BORDER_CONSTANT
    border_value = 0
    L_sight = cv2.copyMakeBorder(Il, border_size, border_size, border_size, border_size, border_type, value=border_value)
    R_sight = cv2.copyMakeBorder(Ir, border_size, border_size, border_size, border_size, border_type, value=border_value)

    # Create kernel
    window_size = 3
    L_binary = np.zeros((window_size**2, *L_sight.shape))
    R_binary = np.zeros((window_size**2, *R_sight.shape))
    idx = 0

    # Scan image with window size = x (kernel)
    for x in range(-(window_size//2), (window_size//2)+1):
        for y in range(-(window_size//2), (window_size//2)+1):
            ''' 大的1/小的0 '''
            maskL = (L_sight > np.roll(L_sight, [y, x], axis=[0, 1]))
            L_binary[idx][maskL] = 1
            maskR = (R_sight > np.roll(R_sight, [y, x], axis=[0, 1]))
            R_binary[idx][maskR] = 1
            idx += 1

    # Depadding
    L_binary = L_binary[:, 1:-1, 1:-1] 
    R_binary = R_binary[:, 1:-1, 1:-1]

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    L_final_cost = np.zeros((max_disp+1, h, w))
    R_final_cost = np.zeros((max_disp+1, h, w))

    # 對每個視差值進行迭代
    for d in range(max_disp+1):
        # 位移左圖和右圖以匹配視差值
        l_shift = L_binary[:, :, d:].astype(np.uint32)
        r_shift = R_binary[:, :, :w-d].astype(np.uint32)
        # 計算Bit差異
        bit_cost = np.sum(np.bitwise_xor(l_shift, r_shift), axis=0).astype(np.float32)
        bit_cost = np.sum(bit_cost, axis=2).astype(np.float32)
        
        # Joint bilateral filter
        l_cost = cv2.copyMakeBorder(bit_cost, 0, 0, d, 0, cv2.BORDER_REPLICATE)
        # print(Il.shape)
        # print(l_cost.shape)
        L_final_cost[d] = xip.jointBilateralFilter(Il, l_cost, -1, 5, 15)
        r_cost = cv2.copyMakeBorder(bit_cost, 0, 0, 0, d, cv2.BORDER_REPLICATE)
        R_final_cost[d] = xip.jointBilateralFilter(Ir, r_cost, -1, 5, 15)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    L_Disparity = np.zeros((h, w), dtype=np.uint8)
    R_Disparity = np.zeros((h, w), dtype=np.uint8)

    # 遍歷圖像的每個像素
    for i in range(h):
        for j in range(w):
            # 在左圖的最終成本中找到最小值的索引作為左圖像素的視差
            L_Disparity[i, j] = np.argmin(L_final_cost[:, i, j])
            # 在右圖的最終成本中找到最小值的索引作為右圖像素的視差
            R_Disparity[i, j] = np.argmin(R_final_cost[:, i, j])
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    '''Left-right consistency check'''
    Left_right_check = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            Rx = x - L_Disparity[y, x]
            if Rx >= 0 and Rx < w:
                r_disp = R_Disparity[y, int(Rx)]
                if L_Disparity[y, x] == r_disp:
                    Left_right_check[y, x] = L_Disparity[y, x]

    '''Hole filling'''
    # Padding first
    Left_right_check_pad = cv2.copyMakeBorder(Left_right_check,0,0,1,1, cv2.BORDER_CONSTANT, value=max_disp)
    L_labels = np.zeros((h, w), dtype=np.float32)
    R_labels = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            idx_L, idx_R = 0, 0
            # 填充左邊的空洞
            while Left_right_check_pad[y, x+1-idx_L] == 0:
                idx_L += 1
            L_labels[y, x] = Left_right_check_pad[y, x+1-idx_L]
            # 填充右邊的空洞
            while Left_right_check_pad[y, x+1+idx_R] == 0:
                idx_R += 1
            R_labels[y, x] = Left_right_check_pad[y, x+1+idx_R]
    
    # Final result
    labels = np.min((L_labels, R_labels), axis=0)

    '''Weighted median filtering'''
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, h//16)
    # labels = xip.guidedFilter(labels, Il.astype(np.uint8), 10, 0.1)
    # labels = cv2.fastNlMeansDenoising(labels.astype(np.uint8), None, 3, 3, 21)
    # print(labels.shape)
    # labels = adaptive_median_filter(labels, 11)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程式運行時間：{elapsed_time:.2f} 秒")

    return labels.astype(np.uint8)
    