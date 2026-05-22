#!/usr/bin/env python3
import sys
sys.path.append('.')

import yaml
import argparse
from src.matchers import FeatureMatcher
from src.kitti_evaluator import KITTIEvaluator

def main():
    parser = argparse.ArgumentParser(description='KITTI Pose 평가')
    parser.add_argument('--config', default='configs/config.yaml')
    args = parser.parse_args()
    
    # Config 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 매칭기 초기화
    print('Initializing matchers...')
    matcher = FeatureMatcher()
    
    # 평가 실행
    evaluator = KITTIEvaluator(config, matcher)
    
    # 사용할 이미지 쌍 정의
    pairs = [
        (0, 1), (0, 5), (0, 10),
        (10, 20), (10, 25),
        (50, 51), (50, 55), (50, 60),
        (100, 101), (100, 110)
    ]
    
    results = evaluator.evaluate(pairs=pairs)
    
    print('\n=== Results DataFrame ===')
    print(results.to_string(index=False))

if __name__ == '__main__':
    main()