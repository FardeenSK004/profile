import cv2
import numpy as np

# Load Haar cascades bundled with opencv-python
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade      = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _resize(img, max_dim=800):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def _detect_faces(gray):
    """Try frontal, then profile cascade. Return list of (x,y,w,h)."""
    faces = frontal_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

    if len(faces) == 0:
        # Fallback for Niqab: If no face detected but clear eyes found, synthesize a face box
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            e1, e2 = eyes[0], eyes[-1]
            x1, y1 = e1[0], min(e1[1], e2[1])
            x2, y2 = e2[0] + e2[2], max(e1[1] + e1[3], e2[1] + e2[3])
            ew, eh = x2 - x1, y2 - y1

            # Synthesize a realistic face bounding box around the eyes
            fw = int(ew * 2.5)
            fh = int(fw * 1.5)
            fx = max(0, x1 - int(ew * 0.75))
            fy = max(0, y1 - int(fh * 0.25))

            faces = [np.array([fx, fy, fw, fh], dtype=np.int32)]

    return list(faces)


def _faces_at_angle(gray, angle):
    """
    Returns number of frontal faces detected in `gray` when rotated by `angle`.
    Uses a slightly looser threshold (minNeighbors=4) to detect faces
    that are clearly present but not upright.
    """
    if angle == 0:
        rotated = gray
    elif angle == 90:
        rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(gray, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated = gray

    faces = frontal_cascade.detectMultiScale(
        rotated, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
    return len(faces)


# Human-readable fix instructions keyed by which rotation corrects the image.
# If face is found at angle X, the user needs to rotate the *other* way to fix it.
_ROTATION_FIX = {
    90:  "rotate it 90° counter-clockwise",
    180: "rotate it 180° (flip it upside-down)",
    270: "rotate it 90° clockwise",
}


def _is_rotated_image(gray):
    """
    Detects if the image content is rotated (90° / 180° / 270°).
    Returns (is_rotated: bool, fix_hint: str).
    An image is only flagged as rotated if all three conditions hold:
      - The best rotated angle finds >= 2 faces (not a spurious single match)
      - The best rotated count is strictly more than 2x the upright count
        OR upright has 0 faces but the rotated angle has faces.
    This prevents low-quality JPEG compression from causing false positives.
    """
    count_0   = _faces_at_angle(gray, 0)
    count_90  = _faces_at_angle(gray, 90)
    count_180 = _faces_at_angle(gray, 180)
    count_270 = _faces_at_angle(gray, 270)

    rotated_counts = {90: count_90, 180: count_180, 270: count_270}
    max_rotated_angle = max(rotated_counts, key=rotated_counts.get)
    max_rotated_count = rotated_counts[max_rotated_angle]

    # Only flag as rotated if:
    # a) Upright finds 0 faces but a rotated angle finds some, OR
    # b) Rotated count is >= 3 AND strictly more than 2x upright count + 1
    #    (requiring a large majority to avoid compression artefact false positives)
    if count_0 == 0 and max_rotated_count > 0:
        is_rotated = True
    elif max_rotated_count >= 3 and max_rotated_count > count_0 * 2 + 1:
        is_rotated = True
    else:
        is_rotated = False

    hint = _ROTATION_FIX.get(max_rotated_angle, "rotate it to be upright")
    return is_rotated, hint



def _skin_ratio(roi_bgr):
    """Fraction of pixels inside YCrCb skin range."""
    ycbcr = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycbcr,
                       np.array([0,  133,  77], np.uint8),
                       np.array([255, 173, 127], np.uint8))
    return np.sum(mask > 0) / max(roi_bgr.shape[0] * roi_bgr.shape[1], 1)


def _color_diversity(roi_bgr):
    """
    Number of distinct quantised Lab colours in a thumbnail.
    Real photos → high diversity; illustrations → low diversity.
    """
    thumb = cv2.resize(roi_bgr, (64, 64))
    lab   = cv2.cvtColor(thumb, cv2.COLOR_BGR2Lab)
    quantised = (lab // 16).reshape(-1, 3)
    return len(set(map(tuple, quantised)))


def _saturation_stats(roi_bgr):
    """Return (mean, std) of HSV saturation channel."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    return s.mean(), s.std()


def _has_unnatural_hair_color(roi_bgr):
    """
    Checks if dominant non-skin hue is in anime/cartoon range:
    vivid blues, purples, greens, or pinks that don't occur in real human hair.
    Real hair is brown (Hue 10-30), black (very low V), dark/auburn.
    Anime hair is often vivid blue (Hue 100-130), purple (Hue 130-160),
    green (Hue 60-80), bright pink (Hue 150-175).
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Focus only on the top 25% of the face ROI where hair tends to be.
    # This prevents red lipstick or clothing in the lower half from triggering it.
    h_crop = int(roi_bgr.shape[0] * 0.25)
    upper_hsv = hsv[:h_crop, :, :]

    hue = upper_hsv[:, :, 0]
    sat = upper_hsv[:, :, 1]
    val = upper_hsv[:, :, 2]

    # Consider only vivid, bright pixels (not dark or washed out)
    vivid_mask = (sat > 80) & (val > 60)
    if vivid_mask.sum() < 50:
        return False

    vivid_hues = hue[vivid_mask]

    # Unnatural hair hues (in OpenCV 0-179 range):
    # Blue: 100–130, Purple: 130–155, Green: 60–85, Hot-pink: 150–175
    unnatural = (
        ((vivid_hues >= 100) & (vivid_hues <= 130)) |   # blue
        ((vivid_hues >= 130) & (vivid_hues <= 155)) |   # purple
        ((vivid_hues >= 60)  & (vivid_hues <= 85))  |   # green
        ((vivid_hues >= 150) & (vivid_hues <= 175))      # pink
    )
    unnatural_ratio = unnatural.sum() / len(vivid_hues)
    return unnatural_ratio > 0.30


def _is_anime_illustration(roi_bgr, is_face_roi=False):
    """
    Dedicated anime/manga/illustration detector combining multiple signals.
    """
    # 1. Unnatural hair colour → strong indicator
    if _has_unnatural_hair_color(roi_bgr):
        return True

    # 2. Very restricted colour palette (flat art)
    diversity = _color_diversity(roi_bgr)
    if diversity < 28:
        return True

    # 3. High Laplacian variance + limited palette = crisp outlines, flat fills
    gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    lap_std = cv2.Laplacian(gray, cv2.CV_64F).std()
    if lap_std > 40 and diversity < 52:
        return True

    # 4. Very high uniform saturation (vivid anime colours)
    #    s_std < 44: real-world outdoor photos with saturated backgrounds have s_std > 38
    #    Flat anime colours tend to have s_std < 30 (very uniform)
    s_mean, s_std = _saturation_stats(roi_bgr)
    if s_mean > 110 and s_std < 30:
        return True

    # 5. Smooth-rendered illustration (Ghibli / Disney style):
    #    This check is only reliable on the full/large image — tiny face ROIs
    #    of real skin have artificially high saturation, causing false positives.
    #    It is intentionally skipped when called on a face crop (is_face_roi=True).
    if not is_face_roi:
        gray    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        lap_std = cv2.Laplacian(gray, cv2.CV_64F).std()
        if lap_std < 28 and s_mean > 95:
            # Niqabs/dark hijabs often trigger this due to low texture + high saturation noise.
            # If we detect clear human eyes on the image, spare it from this rejection.
            eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15))
            if len(eyes) >= 2:
                return False
            # High s_mean from a vivid background (e.g. green leafy background) inflates
            # the score without the image being an illustration — add diversity gate.
            if diversity >= 90:
                return False
            return True

    return False




def _is_placeholder(img_bgr):
    """Detect grey placeholder / silhouette avatars or blank images."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.std() < 20:
        return True
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    if (np.abs(b-g).mean() < 8 and np.abs(g-r).mean() < 8
            and np.abs(b-r).mean() < 8 and gray.std() < 55):
        return True
    return False


def _face_cut_off(x, y, w, h, img_w, img_h, margin=10):
    """
    Returns True if the detected face bounding box is clipped by the image
    boundary — meaning the face is partially outside the frame.
    """
    return (x <= margin or y <= margin or
            x + w >= img_w - margin or y + h >= img_h - margin)


def _large_obstruction(img_bgr, x, y, w, h):
    """
    Checks if there is a large non-skin, non-background region *near* the face
    that likely indicates the person is holding a phone / object or wearing a
    face mask.  We look for a dark or neutral rectangular region of significant
    size overlapping the upper-half of the image near the face area.
    """
    img_h, img_w = img_bgr.shape[:2]

    # Measure non-skin pixels in the face bounding box that are also DARK
    # (a phone screen / body tends to be very dark or very uniform)
    face_roi = img_bgr[y:y + h, x:x + w]
    ycbcr    = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycbcr,
                            np.array([0,  133,  77], np.uint8),
                            np.array([255, 173, 127], np.uint8))
    
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # We care mostly about the UPPER half of the face for true obstructions (phones, etc).
    # The lower half being dark non-skin is very common for Hijabs, Niqabs, or dark collared shirts.
    h_half = h // 2
    upper_gray = gray_face[:h_half, :]
    upper_skin = skin_mask[:h_half, :]
    
    # Dark non-skin pixels in the upper half (e.g. phone held high, large dark sunglasses)
    dark_upper = np.sum((upper_gray < 70) & (upper_skin == 0))
    upper_ratio = dark_upper / max(w * h_half, 1)

    # If more than 35% of the UPPER face bounding box is dark non-skin → object blocking
    return upper_ratio > 0.35


def _check_face_tilt(face_roi_gray):
    """
    Detects eyes inside the face ROI and computes the angle of the line
    connecting them relative to the horizontal axis.
    Returns (is_tilted: bool, angle_degrees: float).
    If no two eyes found, returns (False, 0) to avoid false rejections.
    """
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

    if len(eyes) < 2:
        return False, 0.0

    # Take the two largest detected eyes
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

    # Centre points of each eye
    cx1 = eyes[0][0] + eyes[0][2] // 2
    cy1 = eyes[0][1] + eyes[0][3] // 2
    cx2 = eyes[1][0] + eyes[1][2] // 2
    cy2 = eyes[1][1] + eyes[1][3] // 2

    dx = cx2 - cx1
    dy = cy2 - cy1
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    # arctan2 gives angle relative to horizontal; 0° = perfectly level
    # Angles near 90° mean the face is sideways
    # Normalise to 0-90 range
    if angle > 90:
        angle = 180 - angle

    # Reject if tilt > 25 degrees
    return angle > 25, angle


def _check_image_orientation(img_bgr):
    """
    A profile picture should be portrait (h >= w) or roughly square.
    If the image is significantly wider than tall (landscape), it suggests
    the photo was taken sideways / is a rotated image.
    """
    h, w = img_bgr.shape[:2]
    aspect = w / h
    # Landscape with ratio > 1.3 is suspicious for a profile shot
    return aspect > 1.3



# --------------------------------------------------------------------------- #
#  Main entry-point
# --------------------------------------------------------------------------- #

def analyze_image(image_bytes):
    """
    Analyzes an image to determine if it's a real human face.
    Returns: (is_valid: bool, score: int [0-100], messages: list[str])
    """
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return False, 0, ["Invalid image format or corrupted file."]

        img = _resize(img)
        img_h, img_w = img.shape[:2]

        # ------------------------------------------------------------------ #
        #  0. Whole-image pre-checks
        # ------------------------------------------------------------------ #
        if _is_placeholder(img):
            return False, 0, [
                "Image appears to be blank, black, or a placeholder icon. "
                "Please upload a real photo of yourself."
            ]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ------------------------------------------------------------------ #
        #  0b. Whole-image illustration pre-check
        #  Run this BEFORE the rotation test so anime/cartoon art is caught
        #  with the right reason (the rotation cascade can match on flat art).
        #  EXCEPTION: if frontal cascade already found a face upright, it's likely
        #  a real person — skip this pre-check and rely on face-ROI check instead.
        # ------------------------------------------------------------------ #
        upright_faces = frontal_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        if len(upright_faces) == 0 and _is_anime_illustration(img, is_face_roi=False):
            return False, 0, [
                "Image looks like an illustration, anime, or cartoon artwork. "
                "Profile pictures must be real photographs of the doctor."
            ]

        # ------------------------------------------------------------------ #
        #  1. Rotation check — must be upright (not sideways / upside-down)
        # ------------------------------------------------------------------ #
        is_rotated, rotation_hint = _is_rotated_image(gray)
        if is_rotated:
            return False, 0, [
                f"Your photo is not upright — please {rotation_hint} and re-upload. "
                "Profile pictures must show your face straight-on."
            ]

        # ------------------------------------------------------------------ #
        #  2. Face detection (strict — upright only)
        # ------------------------------------------------------------------ #
        faces = _detect_faces(gray)
        if not faces:
            return False, 0, [
                "No human face detected. Please upload a clear, well-lit photo "
                "of yourself facing the camera."
            ]

        # Work with the largest detected face
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        face_roi = img[y:y + h, x:x + w]

        # Determine if there are clear eyes in the image (used for hijab/niqab bypassing)
        eyes_detected = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15))
        has_clear_eyes = len(eyes_detected) >= 2

        # ------------------------------------------------------------------ #
        #  2. Face must not be clipped at image edge (partial / cut-off face)
        # ------------------------------------------------------------------ #
        # Standard edge-clip check
        if not has_clear_eyes and _face_cut_off(x, y, w, h, img_w, img_h, margin=8):
            return False, 0, [
                "Your face appears to be partially outside the frame. "
                "Please use a photo where your full face is clearly visible."
            ]

        # Extra check: if the top of the detected face is very low in the image
        # relative to image height, the top of the head/forehead is cut off.
        # A proper headshot will have forehead visible near the top ~30% of the image.
        if not has_clear_eyes and y > img_h * 0.45:
            return False, 0, [
                "Your face appears to be partially cut off (top of head missing). "
                "Please use a photo where your full face is clearly visible."
            ]

        # ------------------------------------------------------------------ #
        #  3. Face size check — must be a proper headshot
        # ------------------------------------------------------------------ #
        img_area  = img_w * img_h
        face_area = w * h
        face_ratio = face_area / img_area

        # Niqab / eye-region expansion:
        # If the box is very small, it might just be the eye slit of a Niqab detected as a face.
        # Check the *full image* for clear eyes to bypass overly strict face bounds.
        eyes_detected = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15))
        has_clear_eyes = len(eyes_detected) >= 2

        if has_clear_eyes and face_ratio < 0.08:
            # Artificially expand bounding box to simulate full head
            h_new = min(img_h - y, int(h * 2.5))
            w_new = min(img_w - x, int(w * 1.5))
            x_new = max(0, int(x - w * 0.16))
            y_new = max(0, int(y - h * 0.15))
            x, y, w, h = x_new, y_new, w_new, h_new
            face_roi = img[y:y + h, x:x + w]
            face_area = w * h
            face_ratio = face_area / img_area

        # Hard reject for very small faces: these are scene/group photos or
        # photos where the person is far away — not suitable profile pictures.
        if face_ratio < 0.06:
            return False, 0, [
                "Your face is too small in the image. "
                "Please upload a clear headshot where your face fills most of the frame."
            ]


        score    = 100
        messages = []

        # ------------------------------------------------------------------ #
        #  4. Image orientation — portrait/square only
        # ------------------------------------------------------------------ #
        if _check_image_orientation(img):
            return False, 0, [
                "Image appears to be rotated or in landscape orientation. "
                "Please upload a portrait photo where you are facing straight ahead."
            ]

        # ------------------------------------------------------------------ #
        #  5. Face tilt check — eyes must be roughly horizontal
        # ------------------------------------------------------------------ #
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        is_tilted, tilt_angle = _check_face_tilt(face_gray)
        if is_tilted:
            return False, 0, [
                f"Your face appears to be tilted or rotated "
                f"({int(tilt_angle)}° from horizontal). "
                "Please upload a photo where you are looking straight at the camera."
            ]

        # ------------------------------------------------------------------ #
        #  6. Object / phone obstruction check
        # ------------------------------------------------------------------ #
        # Bypass if we strongly detect two eyes (to allow Hijabs and Niqabs where
        # fabric takes up most of the non-skin face region).
        # Also bypass if the face is an extreme close-up (w > 80% of image),
        # where the "background" is entirely dark clothing.
        is_extreme_closeup = w > img_w * 0.8
        if not has_clear_eyes and not is_extreme_closeup and _large_obstruction(img, x, y, w, h):
            return False, 0, [
                "A large object (e.g. phone, mask) appears to be blocking your face. "
                "Please upload a clear headshot without obstructions."
            ]

        # ------------------------------------------------------------------ #
        #  5. Illustration / anime / cartoon detector
        # ------------------------------------------------------------------ #
        if _is_anime_illustration(face_roi, is_face_roi=True):
            return False, 0, [
                "Image looks like an illustration, anime, or cartoon artwork. "
                "Profile pictures must be real photographs of the doctor."
            ]

        # ------------------------------------------------------------------ #
        #  6. Flat-colour / bitmoji check (inner face crop)
        # ------------------------------------------------------------------ #
        iw = int(w * 0.75); ih = int(h * 0.75)
        ix = (w - iw) // 2; iy = (h - ih) // 2
        inner = face_roi[iy:iy + ih, ix:ix + iw]

        s_mean, s_std = _saturation_stats(inner)
        v_std = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)[:, :, 2].std()

        if s_std < 12 and v_std < 18:
            return False, 0, [
                "Image appears flat-coloured (possible bitmoji or digital avatar). "
                "Please upload a real photograph."
            ]

        # ------------------------------------------------------------------ #
        #  7. Skin tone check
        # ------------------------------------------------------------------ #
        skin = _skin_ratio(inner)
        # Lower skin threshold significantly if it's a Niqab/Hijab (where mostly eyes show)
        min_skin = 0.02 if has_clear_eyes else 0.12
        if skin < min_skin:
            score -= 30
            messages.append(
                "Could not detect natural human skin tones. "
                "Make sure your face is clearly visible and well-lit.")

        # ------------------------------------------------------------------ #
        #  8. Edge density on inner face (texture sanity check)
        # ------------------------------------------------------------------ #
        edges     = cv2.Canny(inner, 80, 160)
        edge_dens = np.sum(edges > 0) / max(iw * ih, 1)

        if edge_dens > 0.28:
            score -= 10
            messages.append(
                "Image appears heavily illustrated or filtered.")
        elif edge_dens < 0.004:
            score -= 10
            messages.append(
                "Image appears unnaturally smooth (possible heavy filter).")

        # ------------------------------------------------------------------ #
        #  Final verdict
        # ------------------------------------------------------------------ #
        is_valid = score >= 60

        if is_valid:
            if not messages:
                messages.append("Great profile photo!")
            else:
                messages.insert(0, "Passed with minor warnings.")
        else:
            messages.insert(0, "Image rejected.")

        return is_valid, max(0, score), messages

    except Exception as exc:
        return False, 0, [f"Error processing image: {exc}"]
