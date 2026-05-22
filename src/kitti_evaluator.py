import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path

class KITTIEvaluator:
    def __init__(self, config, matcher):
        self.config = config
        self.matcher = matcher
        self.K = None
        self.poses = None
    
    def load_calib(self, calib_path):
        """calib.txt에서 P0 읽기"""
        with open(calib_path, 'r') as f:
            for line in f:
                if line.startswith('P0:'):
                    P0 = np.fromstring(line.split(':')[1], sep=' ').reshape(3, 4)
                    return P0[:, :3]  # K matrix
        raise ValueError('P0 not found in calib.txt')
    
    def load_gt_poses(self, poses_path):
        """poses.txt 로드"""
        poses = []
        with open(poses_path, 'r') as f:
            for line in f:
                T = np.fromstring(line, sep=' ').reshape(3, 4)
                T = np.vstack([T, [0, 0, 0, 1]])
                poses.append(T)
        return poses
    
    def get_relative_pose(self, T1, T2):
        """Relative pose 계산"""
        T_rel = np.linalg.inv(T1) @ T2
        R = T_rel[:3, :3]
        t = T_rel[:3, 3]
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        return R, t, angle
    
    def compute_pose_error(self, R_gt, t_gt, R_est, t_est):
        """Pose error 계산"""
        # Rotation error
        R_err = R_gt.T @ R_est
        rot_err = np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))
        
        # Translation error
        t_gt_n = t_gt / np.linalg.norm(t_gt)
        t_est_n = t_est.ravel() / np.linalg.norm(t_est)
        cos_sim = np.dot(t_gt_n, t_est_n)
        t_err = np.degrees(np.arccos(np.clip(abs(cos_sim), -1, 1)))
        
        return rot_err, t_err
    
    def evaluate(self, pairs=None):
        """KITTI 평가 실행"""
        if pairs is None:
            pairs = [(0,1), (0,5), (0,10)]
        
        # 데이터 경로
        data_path = Path(self.config['data']['kitti'])
        images_dir = data_path / 'images'
        poses_file = data_path / 'poses.txt'
        calib_file = data_path / 'calib.txt'
        
        # 로드
        print('Loading KITTI data...')
        self.poses = self.load_gt_poses(poses_file)
        self.K = self.load_calib(calib_file)
        print(f'  ✓ {len(self.poses)} poses')
        print(f'  ✓ K matrix from calib.txt')
        
        # 매칭 방법
        methods = {
            'SP+NN': self.matcher.match_nn,
            'SP+SuperGlue': self.matcher.match_superglue,
            'SP+LightGlue': self.matcher.match_lightglue,
        }
        
        # Warm-up
        print('\nWarm-up...')
        dummy = str(images_dir / '000000.png')
        for _ in range(3):
            self.matcher.match_lightglue(dummy, dummy)
            try:
                self.matcher.match_superglue(dummy, dummy)
            except:
                pass
        print('Done\n')
        
        results = []
        
        print('=== KITTI Evaluation ===\n')
        
        for idx1, idx2 in pairs:
            img1_path = str(images_dir / f'{idx1:06d}.png')
            img2_path = str(images_dir / f'{idx2:06d}.png')
            
            # GT pose
            R_gt, t_gt, angle_gt = self.get_relative_pose(
                self.poses[idx1], self.poses[idx2]
            )
            
            print(f'Pair {idx1:03d}-{idx2:03d} (GT angle: {angle_gt:.2f}°)')
            
            for name, fn in methods.items():
                # 매칭
                times = []
                for _ in range(3):
                    pts1, pts2, t = fn(img1_path, img2_path)
                    times.append(t * 1000)
                t_mean = np.mean(times)
                
                if len(pts1) < 8:
                    print(f'  {name:14s} - Not enough matches: {len(pts1)}')
                    continue
                
                # Essential Matrix
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0
                )
                
                if E is None:
                    print(f'  {name:14s} - Essential matrix failed')
                    continue
                
                # Pose recovery
                _, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, self.K)
                
                # Error 계산
                rot_err, t_err = self.compute_pose_error(
                    R_gt, t_gt, R_est, t_est
                )
                
                inliers = int(mask.sum())
                inlier_ratio = inliers / len(pts1)
                
                results.append({
                    'pair': f'{idx1:03d}-{idx2:03d}',
                    'method': name,
                    'rot_err': round(rot_err, 2),
                    't_err': round(t_err, 2),
                    'matches': len(pts1),
                    'inliers': inliers,
                    'inlier_ratio': round(inlier_ratio, 3),
                    'time_ms': round(t_mean, 2),
                })
                
                print(f'  {name:14s} Rot={rot_err:.2f}° Trans={t_err:.2f}° '
                      f'M={len(pts1)} In={inliers} t={t_mean:.0f}ms')
        
        df = pd.DataFrame(results)
        
        # 평균 출력
        print('\n' + '='*70)
        print('평균 결과')
        print('='*70)
        for method in methods.keys():
            subset = df[df['method'] == method]
            if len(subset) == 0: continue
            print(f'{method:14s} '
                  f'Rot={subset["rot_err"].mean():.2f}° '
                  f'Trans={subset["t_err"].mean():.2f}° '
                  f'IR={subset["inlier_ratio"].mean():.3f} '
                  f't={subset["time_ms"].mean():.0f}ms')
        
        return df