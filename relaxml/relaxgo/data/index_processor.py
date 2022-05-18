# -*- coding:utf-8 -*-
import os
import json
import sys
import multiprocessing
from lxml import etree
import requests
from ..utils import download, download_extract
"""
从KGS围棋服务器上下载自2001年以来棋谱合集. 在这个合集中, 对弈双方要么至少有一方为
7段以上, 要么双方都有6段, 所有的棋局都是在19x19的标准棋盘上进行的
"""


def _worker(url_and_target):
    try:
        url, target_path = url_and_target
        print(f'正在从{url}下载{target_path}...')
        r = requests.get(url, stream=True, verify=True)
        with open(target_path, 'wb') as f:
            f.write(r.content)
        print(f'下载{target_path}完成!')
    except (KeyboardInterrupt, SystemExit):
        print('>>> Exiting child process')


class KGSIndex:
    """
    
    注意: 构造这个类需要联网

    可以从KGS官网或者Aliyun OSS下载棋谱

    下载完的目录结构:
    ../data/
        kgs.json
        kgs.zip
        kgs/
            *.tar.gz
    >>> index = KGSIndex()
    >>> index.download_files()

    """

    def __init__(self,
                 kgs_url: str = 'http://u-go.net/gamerecords/',
                 data_directory: str = '../data') -> None:
        """
        参数:
        kgs_url: KGS URL
        data_directory: 下载数据的保存路径
        """
        self.kgs_url = kgs_url
        self.data_directory = data_directory
        # 文件信息
        # List[Tuple[str, str, int]]
        # [{url, filename, num_games}]
        self.file_info = []
        self.urls = []  # 远程文件URLs
        self.request_headers = {
            'X-Requested-With':
            'XMLHttpRequest',
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/56.0.2924.87 Safari/537.36'
        }

    def load_index(self, src='aliyun') -> None:
        """
        下载文件信息

        参数:
        src: aliyun | kgs 下载源
        """
        if src == 'aliyun':
            f_json = open(download('kgs.json'))
            self.file_info = json.loads(f_json.read())
            for info in self.file_info:
                self.urls.append(info['url'])
        elif src == 'kgs':
            print('Downloading index page...')
            resp = requests.get(self.kgs_url, headers=self.request_headers)
            resp.encoding = 'utf-8'
            data = resp.text
            if not data:
                return None
            html = etree.HTML(data)
            elems = html.xpath('//table[@class="table"]//tr/td/a')
            for elem in elems:
                url = elem.get('href')
                if url.endswith('.tar.gz'):
                    self.urls.append(url)
            for url in self.urls:
                # https://dl.u-go.net/gamerecords/KGS-2019_04-19-1255-.tar.gz
                filename = os.path.basename(url)
                split_file_name = filename.split('-')
                num_games = int(split_file_name[len(split_file_name) - 2])
                # print(f'filename:{filename}, num_games:{num_games}')
                self.file_info.append({
                    'url': url,
                    'filename': filename,
                    'num_games': num_games
                })

    def download_files(self,
                       src='aliyun',
                       multi_process: bool = True,
                       force: bool = False) -> None:
        """
        1. 下载文件信息
        2. 根据文件信息逐个下载文件

        参数:
        src: aliyun | kgs 下载源
        multi_process: 是否启用multi process [kgs]
        force: 是否覆盖本地文件 [kgs]
        """
        if src == 'aliyun':
            # 下载: kgs.json
            self.load_index(src)
            # 下载: kgs.zip
            download_extract('kgs', self.data_directory)
        elif src == 'kgs':
            if not self.urls:
                self.load_index(src)

            if not os.path.isdir(self.data_directory):
                os.makedirs(self.data_directory)
                os.makedirs(os.path.join(self.data_directory, 'kgs'))

            urls_to_download = []
            for file_info in self.file_info:
                url = file_info['url']
                file_name = file_info['filename']
                # 对本地不存在的文件重新下载, 已经存在的忽略
                if force or not os.path.isfile(self.data_directory + '/kgs/' +
                                               file_name):
                    urls_to_download.append(
                        (url, self.data_directory + '/kgs/' + file_name))

            if multi_process:
                cores = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(processes=cores)
                try:
                    it = pool.imap(_worker, urls_to_download)
                    for _ in it:
                        pass
                    pool.close()
                    pool.join()
                except KeyboardInterrupt:
                    print(">>> Caught KeyboardInterrupt, terminating workers")
                    pool.terminate()
                    pool.join()
                    sys.exit(-1)
            else:
                for url_and_target in urls_to_download:
                    _worker(url_and_target)
