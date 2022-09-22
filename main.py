from multiprocessing import Process
from broker.main_img_func import process_images
from broker.functions import receive
from broker.pika_consts import broker_password_img_work, broker_user_img_work, main_queue_name
from ya_disk.ya_disk_main import ya_main


def main_loop():
    receive(main_queue_name, user=broker_user_img_work, password=broker_password_img_work,
            do_function=process_images, initArgs='main')


def ya_disk_loop():
    ya_main()


def main():
    procs = []
    proc_main = Process(target=main_loop)  # instantiating without any argument
    procs.append(proc_main)
    proc_main.start()
    proc_yadisk = Process(target=ya_disk_loop)  # instantiating without any argument
    procs.append(proc_yadisk)
    proc_yadisk.start()

    # complete the processes
    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main()

