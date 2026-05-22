import cv2
import time
import torch
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

class FeatureMatcher:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')
        
        # SuperPoint 추출기
        self.extractor_sp = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        
        # LightGlue 매칭기
        self.matcher_lg = LightGlue(features='superpoint').eval().to(self.device)
        
        # SuperGlue 매칭기
        try:
            import sys
            sys.path.append('./models/SuperGlue')
            from models.matching import Matching
            
            superglue_config = {
                'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 2048},
                'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2},
            }
            self.matching_sg = Matching(superglue_config).eval().to(self.device)
            print('SuperGlue loaded')
        except Exception as e:
            print(f"Warning: SuperGlue not available - {e}")
            self.matching_sg = None
    
    def match_nn(self, img1_path, img2_path):
        img1_orig = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_orig = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        scale1 = 640.0 / max(img1_orig.shape[:2])
        scale2 = 640.0 / max(img2_orig.shape[:2])
        img1_scaled = cv2.resize(img1_orig, (int(img1_orig.shape[1]*scale1), int(img1_orig.shape[0]*scale1)))
        img2_scaled = cv2.resize(img2_orig, (int(img2_orig.shape[1]*scale2), int(img2_orig.shape[0]*scale2)))
        
        t1 = torch.from_numpy(img1_scaled/255.).float()[None, None].to(self.device)
        t2 = torch.from_numpy(img2_scaled/255.).float()[None, None].to(self.device)
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.no_grad():
            f0 = self.extractor_sp({'image': t1})
            f1 = self.extractor_sp({'image': t2})
            f0, f1 = rbd(f0), rbd(f1)
        
        kp0 = f0['keypoints'].cpu().numpy()
        kp1 = f1['keypoints'].cpu().numpy()
        desc0 = f0['descriptors'].cpu().numpy()
        desc1 = f1['descriptors'].cpu().numpy()
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(np.ascontiguousarray(desc0.T, dtype=np.float32), np.ascontiguousarray(desc1.T, dtype=np.float32))
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        if len(matches) == 0: return np.zeros((0,2)), np.zeros((0,2)), elapsed
        
        # 💡 640px 해상도 좌표 그대로 반환
        pts1 = np.float32([kp0[m.queryIdx] for m in matches])
        pts2 = np.float32([kp1[m.trainIdx] for m in matches])
        return pts1, pts2, elapsed

    def match_superglue(self, img1_path, img2_path):
        img1_orig = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_orig = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        scale1 = 640.0 / max(img1_orig.shape[:2])
        scale2 = 640.0 / max(img2_orig.shape[:2])
        img1_scaled = cv2.resize(img1_orig, (int(img1_orig.shape[1]*scale1), int(img1_orig.shape[0]*scale1)))
        img2_scaled = cv2.resize(img2_orig, (int(img2_orig.shape[1]*scale2), int(img2_orig.shape[0]*scale2)))
        
        t1 = torch.from_numpy(img1_scaled/255.).float()[None, None].to(self.device)
        t2 = torch.from_numpy(img2_scaled/255.).float()[None, None].to(self.device)
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.no_grad():
            pred = self.matching_sg({'image0': t1, 'image1': t2})
            
        if self.device.type == 'cuda': torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        kp0 = pred['keypoints0'][0].cpu().numpy()
        kp1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        
        valid = matches > -1
        # 💡 640px 해상도 좌표 그대로 반환
        return kp0[valid], kp1[matches[valid]], elapsed

    def match_lightglue(self, img1_path, img2_path):
        img1_orig = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_orig = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        
        scale1 = 640.0 / max(img1_orig.shape[:2])
        scale2 = 640.0 / max(img2_orig.shape[:2])
        img1_scaled = cv2.resize(img1_orig, (int(img1_orig.shape[1]*scale1), int(img1_orig.shape[0]*scale1)))
        img2_scaled = cv2.resize(img2_orig, (int(img2_orig.shape[1]*scale2), int(img2_orig.shape[0]*scale2)))
        
        t1 = torch.from_numpy(img1_scaled/255.).float()[None, None].to(self.device)
        t2 = torch.from_numpy(img2_scaled/255.).float()[None, None].to(self.device)
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        
        with torch.no_grad():
            f0 = self.extractor_sp({'image': t1})
            f1 = self.extractor_sp({'image': t2})
            m01 = self.matcher_lg({'image0': f0, 'image1': f1})
            f0, f1, m01 = [rbd(x) for x in [f0, f1, m01]]
            
        if self.device.type == 'cuda': torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        mt = m01['matches']
        # 💡 640px 해상도 좌표 그대로 반환
        pts1 = f0['keypoints'][mt[...,0]].cpu().numpy()
        pts2 = f1['keypoints'][mt[...,1]].cpu().numpy()
        return pts1, pts2, elapsed
