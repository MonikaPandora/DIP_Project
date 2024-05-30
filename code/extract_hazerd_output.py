import os
import json
import re


def extract_HazeRD_matlab_metrics(filepath):
    res = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            match = re.match(r'Evaluating.*_([0-9]+).', lines[i])
            if not match:
                break
            see = int(match.groups()[0])

            res.setdefault(see, {})
            res[see].setdefault('ssim', [])
            res[see].setdefault('ciede', [])
            res[see]['ssim'].append(float(lines[i + 1].split()[1]))
            res[see]['ciede'].append(float(lines[i + 2].split()[1]))

            i += 3
    for see in res:
        res[see]['ssim'] = sum(res[see]['ssim']) / len(res[see]['ssim'])
        res[see]['ciede'] = sum(res[see]['ciede']) / len(res[see]['ciede'])
    return dict(sorted(res.items(), key=lambda x: x[0]))


def extract_HazeRD_python_metrics(filepath):
    res = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in list(f):
            match = re.match(
                r'.*\[.+_.+_([0-9]+)\].*(MSE [0-9]+\.[0-9]+).*(PSNR [0-9]+\.[0-9]+).*(SSIM [0-9]+\.[0-9]+).*(CIEDE2000: [0-9]+\.[0-9]+).*',
                line)

            if not match:
                continue
            groups = list(match.groups())
            see, mse, psnr, ssim, ciede = groups
            see = int(see)
            res.setdefault(see, {})
            res[see].setdefault('mse', [])
            res[see].setdefault('psnr', [])
            res[see].setdefault('ssim', [])
            res[see].setdefault('ciede', [])
            res[see]['mse'].append(float(mse.split()[1]))
            res[see]['psnr'].append(float(psnr.split()[1]))
            res[see]['ssim'].append(float(ssim.split()[1]))
            res[see]['ciede'].append(float(ciede.split()[1]))
    for see in res:
        res[see]['mse'] = sum(res[see]['mse']) / len(res[see]['mse'])
        res[see]['psnr'] = sum(res[see]['psnr']) / len(res[see]['psnr'])
        res[see]['ssim'] = sum(res[see]['ssim']) / len(res[see]['ssim'])
        res[see]['ciede'] = sum(res[see]['ciede']) / len(res[see]['ciede'])
    return dict(sorted(res.items(), key=lambda x: x[0]))


if __name__ == '__main__':
    os.makedirs('../metrics/enhanced', exist_ok=True)
    res_python = extract_HazeRD_python_metrics('../enhanced_results/test_enhanced_its_output.txt')
    res_matlab = extract_HazeRD_matlab_metrics('../enhanced_results/test_enhanced_its_matlab.txt')

    for see in res_python:
        res_python[see]['ssim'] = res_matlab[see]['ssim']
        res_python[see]['ciede'] = res_matlab[see]['ciede']
    with open('../metrics/enhanced/HazeRD_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(res_python, f, ensure_ascii=True)
