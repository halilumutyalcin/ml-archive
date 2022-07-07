import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

pickle_in = open("camera_cal/wide_dist_pickle.p", "rb")
camera_calibration = pickle.load(pickle_in)
mtx = camera_calibration['mtx']
dist = camera_calibration['dist']

pickle_in = open("camera_cal/warp_pickle.p", "rb")
warp_matrixes = pickle.load(pickle_in)
M = warp_matrixes['m']
M_inv = warp_matrixes['m_inv']
print(M)
print(M_inv)

def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(50, 100)):

    if (orient == 'x'):
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sbinary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
 
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel = pow((pow(sobelX, 2) + pow(sobelY, 2)), 1 / 2)

    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))

    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return sbinary


def dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3)):
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelX)
    abs_sobely = np.absolute(sobelY)

    dir_gradient = np.arctan2(abs_sobely, abs_sobelx)

    sbinary = np.zeros_like(dir_gradient)
    sbinary[(dir_gradient >= thresh[0]) & (dir_gradient <= thresh[1])] = 1

    return sbinary


def color_threshold(img, s_thresh=(170, 255)):
    s_binary = np.zeros_like(img)
    s_binary[(img >= s_thresh[0]) & (img <= s_thresh[1])] = 1
    return s_binary

def get_warped_binary_V(img):
    img_size = (img.shape[1], img.shape[0])

    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    hsv_img = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HSV)
    V_img = hsv_img[:, :, 2]
    V_img_blur = cv2.GaussianBlur(V_img, (5, 5), 0)

    V_sobel_X = abs_sobel_thresh(V_img, orient='x', thresh=(30, 100))
    V_sobel_Y = abs_sobel_thresh(V_img, orient='y', thresh=(30, 100))

    V_sobel_XY = mag_thresh(V_img, mag_thresh=(30, 100))

    V_sobel_Dir = dir_threshold(V_img, thresh=(0.5, 1.3))

    V_comb = np.zeros_like(V_img)
    V_comb[((V_sobel_X == 1) & (V_sobel_Y == 1)) | ((V_sobel_XY == 1) & (V_sobel_Dir == 1))] = 1

    V_col = color_threshold(V_img, s_thresh=(215, 255))

    V_combined_2 = np.zeros_like(V_img)
    V_combined_2[(V_comb == 1) | (V_col == 1)] = 1

    warped_binary2 = cv2.warpPerspective(V_combined_2, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_binary2


def get_warped_binary_S(img):
    img_size = (img.shape[1], img.shape[0])

    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

    hls_img = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HLS)
    S_img = hls_img[:, :, 2]

    S_sobel_X = abs_sobel_thresh(S_img, orient='x', thresh=(10, 100))
    S_sobel_Y = abs_sobel_thresh(S_img, orient='y', thresh=(10, 100))

    S_sobel_XY = mag_thresh(S_img, mag_thresh=(10, 200))

    S_sobel_Dir = dir_threshold(S_img, thresh=(0.5, 1.3))

    S_comb = np.zeros_like(S_img)
    S_comb[((S_sobel_X == 1) & (S_sobel_Y == 1)) | ((S_sobel_XY == 1) & (S_sobel_Dir == 1))] = 1

    S_col = color_threshold(S_img, s_thresh=(170, 255))

    S_combined_2 = np.zeros_like(S_img)
    S_combined_2[(S_comb == 1) | (S_col == 1)] = 1
    S_combined2 = S_comb
    V_combined_2 = get_warped_binary_V(img)

    warped_binary2 = cv2.warpPerspective(S_combined_2, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_binary2


def get_warped_binary_R(img):
    img_size = (img.shape[1], img.shape[0])
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    R_img = img[:, :, 0]

    R_sobel_X = abs_sobel_thresh(R_img, orient='x', thresh=(20, 100))
    R_sobel_Y = abs_sobel_thresh(R_img, orient='y', thresh=(20, 100))

    R_sobel_XY = mag_thresh(R_img, mag_thresh=(15, 200))

    R_sobel_Dir = dir_threshold(R_img, thresh=(0.5, 1.3))

    R_comb = np.zeros_like(R_img)
    R_comb[((R_sobel_X == 1) & (R_sobel_Y == 1)) | ((R_sobel_XY == 1) & (R_sobel_Dir == 1))] = 1

    R_col = color_threshold(R_img, s_thresh=(220, 255))

    R_combined_2 = np.zeros_like(R_img)
    R_combined_2[(R_comb == 1) | (R_col == 1)] = 1
    R_combined2 = R_comb

    warped_binary2 = cv2.warpPerspective(R_combined_2, M, img_size, flags=cv2.INTER_LINEAR)

    V_warped = get_warped_binary_V(img)
    V_warped = get_warped_binary_V(img)
    out = np.zeros_like(R_img)

    out[(warped_binary2 == 1) | (V_warped == 1)] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return opening



def find_lane_pixels(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    dead_zone_x = histogram.shape[0] // 6
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[dead_zone_x:midpoint]) + dead_zone_x
    rightx_base = np.argmax(histogram[midpoint:histogram.shape[0] - dead_zone_x]) + midpoint

    nwindows = 15

    margin = 150

    minpix = 500
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = int(leftx_current - margin / 2)  # Update this
        win_xleft_high = int(leftx_current + margin / 2)  # Update this
        win_xright_low = int(rightx_current - margin / 2)  # Update this
        win_xright_high = int(rightx_current + margin / 2)  # Update this

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if (len(good_left_inds) > minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, drawEnable=False):
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

    cv2.polylines(out_img, np.int32(pts_left), False, (0, 255, 0), thickness=5)
    cv2.polylines(out_img, np.int32(pts_right), False, (0, 255, 0), thickness=5)
    if (drawEnable):
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty, left_fit, right_fit


def search_around_poly(binary_warped, left_fit, right_fit, drawEnable=False):
    margin = 90

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    if (drawEnable):
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')


    return result, left_fit, right_fit, ploty


def draw_area(img, left_fit, right_fit, ploty, locked):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    warp_zero = np.zeros_like(gray).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    pts_left_line = np.vstack((left_fitx, ploty)).astype(np.int32).T
    pts_right_line = np.vstack((right_fitx, ploty)).astype(np.int32).T
    cv2.polylines(color_warp, np.int32(pts_left), False, (255, 0, 0), thickness=20)
    cv2.polylines(color_warp, np.int32(pts_right), False, (0, 0, 255), thickness=20)

    if (locked):
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    else:
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))


    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    combined = cv2.addWeighted(warped, 1, color_warp, 0.8, 0)

    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))

    result = cv2.addWeighted(undist, 1, newwarp, 0.8, 0)
    return combined, result


class Lines():
    def __init__(self):

        self.detected = False

        self.recent_xfitted = []

        self.bestx = None

        self.best_fit = np.array([0, 0, 0], dtype='int')
        self.last_best_fit = np.array([0, 0, 0], dtype='int')

        self.current_fit = [np.array([False])]

        self.radius_of_curvature = 0.0

        self.line_base_pos = None

        self.diffs = np.array([0, 0, 0], dtype='float')

        self.allx = None

        self.ally = None
        self.ploty = np.array([0, 0, 0], dtype='int')

        self.filter_window = None
        self.measures = np.empty((0, 3), int)
        self.wrong_count = 0

    def measure_curvature_real(self, actual_fit, ploty):
        ym_per_pix = 40 / 700  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        fitx = actual_fit[0] * ploty ** 2 + actual_fit[1] * ploty + actual_fit[2]

        fit = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

        y_eval = np.max(ploty) * ym_per_pix
        
        return curvature * fit[0] / abs(fit[0])


    def add_measure(self, measure, ploty):

        self.ploty = ploty
        if (self.radius_of_curvature == 0):
            actual_curvature = self.measure_curvature_real(measure, ploty)
            self.radius_of_curvature = actual_curvature
            self.measures = np.append(self.measures, np.array([measure]), axis=0)
        else:

            self.measures = np.append(self.measures, np.array([measure]), axis=0)
            if self.measures.shape[0] > 4:
                self.measures = np.delete(self.measures, (0), axis=0)
        self.best_fit = np.average(self.measures, axis=0)


        if (self.best_fit[0] == 0):
            self.radius_of_curvature = self.measure_curvature_real(self.last_best_fit, ploty)
        else:
            self.radius_of_curvature = self.measure_curvature_real(self.best_fit, ploty)


lane_left = Lines()
lane_right = Lines()
curvature = 0.0
position = 0.0

def horizontal_distance(left_fit, right_fit, ploty):
    xm_per_pix = 3.7 / 700
    left_fitx = (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]) * xm_per_pix
    right_fitx = (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]) * xm_per_pix
    average_distance = np.average(right_fitx - left_fitx)
    std_distance = np.std(right_fitx - left_fitx)

    x_der = right_fitx[right_fitx.shape[0] - 1]
    x_izq = left_fitx[left_fitx.shape[0] - 1]
    center_car = (1280 * xm_per_pix / 2.0)
    center_road = ((x_der + x_izq) / 2.0)
    position = center_car - center_road
    return average_distance, std_distance, position


def measure_curvature_real(actual_fit, ploty):
    '''
    Calculates the curvature o
    f polynomial functions in meters.
    '''
    ym_per_pix = 40 / 700
    xm_per_pix = 3.7 / 700

    fitx = actual_fit[0] * ploty ** 2 + actual_fit[1] * ploty + actual_fit[2]

    fit = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)

    y_eval = np.max(ploty) * ym_per_pix

    curvature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
    return curvature * fit[0] / abs(fit[0])

def advanced_process_image(img):
    global lane_left
    global lane_right
    global curvature
    global car_position
    alpha = 0.9
    error = False
    print("==================================")

    if (lane_right.wrong_count >= 2):
        print("right:" + str(lane_right.best_fit))
        if (lane_right.best_fit[0] != 0):
            lane_right.last_best_fit = lane_right.best_fit
        lane_right.best_fit = np.array([0, 0, 0], dtype='int')
        lane_right.radius_of_curvature = 0.0
        lane_right.measures = np.empty((0, 3), int)
        lane_right.detected = False

    if (lane_left.wrong_count >= 2):
        print("left" + str(lane_left.best_fit))
        if (lane_left.best_fit[0] != 0):
            lane_left.last_best_fit = lane_left.best_fit
        lane_left.best_fit = np.array([0, 0, 0], dtype='int')
        lane_left.radius_of_curvature = 0.0
        lane_left.measures = np.empty((0, 3), int)
        lane_left.detected = False

    binary_warped = get_warped_binary_R(img)
  
    if (lane_left.detected and lane_right.detected):
        out, left_fit, right_fit, ploty = search_around_poly(binary_warped, lane_left.best_fit, lane_right.best_fit)
    else:
        out, left_fit, right_fit, ploty = fit_polynomial(binary_warped)
        lane_left.detected = True
        lane_right.detected = True

    actual_left_curvature = measure_curvature_real(left_fit, ploty)
    actual_right_curvature = measure_curvature_real(right_fit, ploty)
    h_distance_avg, h_distancd_std, position = horizontal_distance(left_fit, right_fit, ploty)

    if (((h_distance_avg > 3.0) and (h_distance_avg < 4.4) and (h_distancd_std < 0.23)) and (
            (actual_left_curvature / actual_right_curvature > 0) or (abs(actual_left_curvature) > 5000 or abs(
            actual_right_curvature) > 5000))):  
        if ((
                abs(actual_left_curvature / lane_left.radius_of_curvature) < 3.0) or lane_left.radius_of_curvature == 0.0):  # Big difference of curvature
            lane_left.add_measure(left_fit, ploty)
            lane_left.wrong_count = 0
        else:
            lane_left.wrong_count = lane_left.wrong_count + 1
            print("left curv:" + str(actual_left_curvature))

        if ((
                abs(actual_right_curvature / lane_right.radius_of_curvature) < 3.0) or lane_right.radius_of_curvature == 0.0):  
            lane_right.add_measure(right_fit, ploty)
            lane_right.wrong_count = 0
        else:
            lane_right.wrong_count = lane_right.wrong_count + 1
            print("right_curv:" + str(actual_right_curvature))
    else:
        lane_left.wrong_count = lane_left.wrong_count + 1
        lane_right.wrong_count = lane_right.wrong_count + 1
        error = True

    locked = (lane_left.best_fit[0] != 0) or (lane_right.best_fit[0] != 0)

    if (lane_left.best_fit[0] == 0):
        if (lane_right.best_fit[0] == 0):
            color_warped, result = draw_area(img, lane_left.last_best_fit, lane_right.last_best_fit, ploty, locked)
        else:
            color_warped, result = draw_area(img, lane_left.last_best_fit, lane_right.best_fit, ploty, locked)
    else:
        if (lane_right.best_fit[0] == 0):
            color_warped, result = draw_area(img, lane_left.best_fit, lane_right.last_best_fit, ploty, locked)
        else:
            color_warped, result = draw_area(img, lane_left.best_fit, lane_right.best_fit, ploty, locked)

    if (curvature != 0.0):
        curvature = (1 - alpha) * ((abs(lane_left.radius_of_curvature) + abs(
            lane_right.radius_of_curvature)) / 2.0) + alpha * curvature
        car_position = (1 - alpha) * position + alpha * car_position
    else:
        curvature = ((abs(lane_left.radius_of_curvature) + abs(lane_right.radius_of_curvature)) / 2.0)
        car_position = position

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Radius of curvature: " + "{:.2f}".format(curvature) + "(m)", (20, 70), font, 1.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    if (car_position > 0):
        cv2.putText(result, "Car is: " + "{:.2f}".format(abs(car_position)) + "m right of center", (20, 150), font, 1.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result, "Car is: " + "{:.2f}".format(abs(car_position)) + "m left of center", (20, 150), font, 1.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
    if (locked):
        cv2.putText(result, "Target locked", (20, 270), font, 2.0, (0, 255, 0), 4, cv2.LINE_AA)
    else:
        cv2.putText(result, "Target lost", (20, 270), font, 2.0, (0, 0, 255), 4, cv2.LINE_AA)
        
    return color_warped, out, binary_warped, result, error


fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter("out.avi", fourcc, 25.0, (1280, 720))
cap = cv2.VideoCapture('project_video.mp4')

while (cap.isOpened()):
    ret, frame = cap.read()
    color_warped, windows, binary_warped, result, error = advanced_process_image(frame)
    if (error):
        cv2.imwrite("error.jpg", frame)

    windows = cv2.resize(windows, (0, 0), None, .3, .3)
    color_warped = cv2.resize(color_warped, (0, 0), None, .3, .3)
    binary_warped = cv2.resize(255 * binary_warped, (0, 0), None, .47, .47)
    
    x_offset = y_offset = 50
    result[5:5 + color_warped.shape[0], 890:890 + color_warped.shape[1]] = color_warped
    result[230:230 + windows.shape[0], 890:890 + windows.shape[1]] = windows

    out.write(result)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
    else:
        if key == ord('p'):
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', result)
                if key2 == ord('p'):
                    break
        cv2.imshow('frame', result)

cap.release()
cv2.destroyAllWindows()
