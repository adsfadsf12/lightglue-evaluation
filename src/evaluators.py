import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class HomographyEvaluator:
    def __init__(self, config, matcher):
        self.config = config
        self.matcher = matcher
        self.TARGET_SIZE = config['evaluation']['resize_to']
    
    def evaluate(self, dataset='v_poster', show_images=False):
        dataset_path = Path(self.config['data'][dataset])
        
        methods = {
            'SP+NN': self.matcher.match_nn,
            'SP+SuperGlue': self.matcher.match_superglue,
            'SP+LightGlue': self.matcher.match_lightglue,
        }
        
        # Warm-up
        print('Warm-up 중...')
        dummy_img = str(list(dataset_path.rglob('*.jpg'))[0])
        for _ in range(3):
            self.matcher.match_lightglue(dummy_img, dummy_img)
            try:
                self.matcher.match_superglue(dummy_img, dummy_img)
            except:
                pass
        print('Warm-up 완료\n')
        
        results = []
        
        # 이미지 1 찾기
        img1_path = self._find_image(dataset_path, '1')
        if img1_path is None:
            print(f'Error: 1.jpg not found in {dataset_path}')
            return pd.DataFrame()
        
        orig_img1 = cv2.imread(str(img1_path))
        oh1, ow1 = orig_img1.shape[:2]
        scale1 = self.TARGET_SIZE / min(oh1, ow1)
        rh1 = int(oh1 * scale1)
        rw1 = int(ow1 * scale1)
        
        tmp1 = '/tmp/img1.jpg'
        cv2.imwrite(tmp1, cv2.resize(orig_img1, (rw1, rh1)))
        
        category = 'illumination' if 'illum' in dataset else 'viewpoint'
        print(f'=== {dataset} ({category}) ===\n')
        
        for idx in range(2, 7):
            img2_path = self._find_image(dataset_path, str(idx))
            if img2_path is None: continue
            
            orig_img2 = cv2.imread(str(img2_path))
            oh2, ow2 = orig_img2.shape[:2]
            scale2 = self.TARGET_SIZE / min(oh2, ow2)
            rh2 = int(oh2 * scale2)
            rw2 = int(ow2 * scale2)
            
            tmp2 = '/tmp/img2.jpg'
            cv2.imwrite(tmp2, cv2.resize(orig_img2, (rw2, rh2)))
            
            # H_gt 로드
            h_path = dataset_path / f'H_1_{idx}'
            if h_path.exists():
                H_gt = np.loadtxt(str(h_path))
            elif category == 'illumination':
                H_gt = np.eye(3)
            else:
                continue
            
            # 스케일 변환
            S1 = np.diag([scale1, scale1, 1.0])
            S2 = np.diag([scale2, scale2, 1.0])
            H_gt_scaled = S2 @ H_gt @ np.linalg.inv(S1)
            
            for name, fn in methods.items():
                times = []
                for _ in range(3):
                    pts1, pts2, t = fn(tmp1, tmp2)
                    times.append(t * 1000)
                t_mean = np.mean(times)
                
                prec, rec = self._compute_precision_recall(pts1, pts2, H_gt_scaled)
                hom = self._compute_homography_auc(pts1, pts2, H_gt_scaled, (rh1, rw1))
                
                results.append({
                    'pair': f'1-{idx}',
                    'method': name,
                    'precision': round(prec, 3),
                    'recall': round(rec, 3),
                    'ransac_auc5': hom['ransac_auc5'],
                    'time_ms': round(t_mean, 2),
                })
                
                print(f'  1-{idx} {name:14s} P={prec:.3f} R={rec:.3f} R@5={hom["ransac_auc5"]:.0f} t={t_mean:.0f}ms')
        
        df = pd.DataFrame(results)
        
        # 평균 출력
        print('\n' + '='*70)
        print(f'{dataset} 평균')
        print('='*70)
        for method in methods.keys():
            subset = df[df['method'] == method]
            if len(subset) == 0: continue
            print(f'{method:14s} P={subset["precision"].mean():.3f} '
                  f'R={subset["recall"].mean():.3f} '
                  f'R@5={subset["ransac_auc5"].mean():.2f} '
                  f't={subset["time_ms"].mean():.0f}ms')
        
        return df
    
    def _find_image(self, folder, base):
        for ext in ['.jpg', '.jpeg', '.png']:
            p = folder / f'{base}{ext}'
            if p.exists(): return p
        return None
    
    def _compute_precision_recall(self, pts1, pts2, H_gt, threshold=5.0):
        if len(pts1) == 0: return 0.0, 0.0
        ones = np.ones((len(pts1), 1))
        pts1_h = np.hstack([pts1, ones])
        pts1_proj = (H_gt @ pts1_h.T).T
        pts1_proj = pts1_proj[:, :2] / (pts1_proj[:, 2:3] + 1e-9)
        errors = np.linalg.norm(pts2 - pts1_proj, axis=1)
        inliers = np.sum(errors < threshold)
        return inliers / len(pts1), inliers / len(pts1)
    
    def _compute_homography_auc(self, pts1, pts2, H_gt, img_shape):
        res = {'ransac_auc5': 0.0}
        if len(pts1) < 4: return res
        
        h, w = img_shape[:2]
        corners = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
        corners_h = np.hstack([corners, np.ones((4,1))])
        corners_gt = (H_gt @ corners_h.T).T
        corners_gt = corners_gt[:, :2] / (corners_gt[:, 2:3] + 1e-9)
        
        H_est, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H_est is not None:
            corners_est = (H_est @ corners_h.T).T
            corners_est = corners_est[:, :2] / (corners_est[:, 2:3] + 1e-9)
            err = np.mean(np.linalg.norm(corners_gt - corners_est, axis=1))
            res['ransac_auc5'] = 1.0 if err <= 5.0 else 0.0
        
        return res