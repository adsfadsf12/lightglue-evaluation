#!/usr/bin/env python3
import sys
sys.path.append('.')

import yaml
import argparse
from src.matchers import FeatureMatcher
from src.evaluators import HomographyEvaluator

def main():
    parser = argparse.ArgumentParser(description='Homography 평가')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--dataset', choices=['v_poster', 'illumination'], required=True, help='Dataset to evaluate')
    parser.add_argument('--show-images', action='store_true', help='Show visualization')
    args = parser.parse_args()
    
    # Config 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # 매칭기 초기화
    print('Initializing matchers...')
    matcher = FeatureMatcher()
    
    # 평가 실행
    evaluator = HomographyEvaluator(config, matcher)
    results = evaluator.evaluate(
        dataset=args.dataset,
        show_images=args.show_images
    )
    
    print('\n=== Results DataFrame ===')
    print(results.to_string(index=False))

if __name__ == '__main__':
    main()