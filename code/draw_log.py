import matplotlib.pyplot as plt
import re


def extract(file_path):
    ret = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        logs = [re.findall(r'\[loss.+?\s[0-9]*\.[0-9]*\]', line) for line in lines]
        for log in logs:
            for pair in log:
                k, v = pair[1:-1].split()
                if k in ret:
                    ret[k].append(float(v))
                else:
                    ret[k] = [float(v)]
    return ret


if __name__ == '__main__':
    d = extract('../origin_results/log/train_its_log.txt')
    x = list(range(len(list(d.items())[0][1])))
    for k, v in d.items():
        if k in ['loss_t', 'loss_a']:
            continue
        plt.plot(x, v, label=k)
    plt.title('Train loss of ITS on baseline')
    plt.xlabel('per 100 steps')
    plt.ylabel('loss')
    # plt.legend(bbox_to_anchor=(0.0, 0.4), loc='upper left')
    plt.legend()
    plt.show()
