import script

import multiprocessing

def f(filename):
    script.SeparatePage().treat_file(filename)

#script.SeparatePage().treat_file("tests/0001.png")
if __name__ == '__main__':
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        multiple_results = [pool.apply_async(f, args=["tests/0001.png"])]
        [res.get(timeout=10000) for res in multiple_results]
