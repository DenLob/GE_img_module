from broker.functions import receive
from broker.pika_consts import queue_ya_disk, broker_password_ya_disk, broker_user_ya_disk
from ya_disk.functions import copy_to_yadisk


def ya_main():
    receive(queue_name=queue_ya_disk, user=broker_user_ya_disk, password=broker_password_ya_disk,
            do_function=copy_to_yadisk, initArgs='ya')