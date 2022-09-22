import functools
import threading

from memory_profiler import memory_usage
from retry import retry
import pika
from pika.exceptions import ConnectionWrongStateError, StreamLostError, ChannelClosedByBroker, AMQPConnectionError
from pika.adapters.utils.connection_workflow import AMQPConnectorStackTimeout

from constants import broker_ip, broker_port
from log_dir.loggers import setup_logger


def ack_message(ch, delivery_tag):
    """Note that `ch` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if ch.is_open:
        ch.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def do_work(conn, ch, delivery_tag, body, do_function, logger):
    thread_id = threading.get_ident()
    logger.info(f"Body={body}, thread_id={thread_id}")
    do_function(body.decode('utf-8'), logger)
    cb = functools.partial(ack_message, ch, delivery_tag)
    conn.add_callback_threadsafe(cb)


def on_message(ch, method_frame, _header_frame, body, args):
    (conn, thrds, do_function, initArgs) = args
    logger = setup_logger(initArgs, initArgs)
    delivery_tag = method_frame.delivery_tag
    logger.info('Before creating new thread ' + ''.join(map(str, memory_usage())))

    t = threading.Thread(target=do_work,
                         args=(conn, ch, delivery_tag, body, do_function, logger))
    t.start()
    thrds.append(t)
    logger.debug(f'Active number of threads are {threading.active_count()}')


@retry(exceptions=(
        ConnectionWrongStateError, StreamLostError, ChannelClosedByBroker, AMQPConnectionError,
        AMQPConnectorStackTimeout),
    tries=-1, delay=300)
def receive(queue_name, user, password, do_function, initArgs=None):
    credentials = pika.PlainCredentials(user, password)
    parameters = pika.ConnectionParameters(broker_ip,
                                           broker_port,
                                           '/',
                                           credentials,
                                           heartbeat=60)

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()

    queue = channel.queue_declare(queue=queue_name,  # Название очереди
                                  durable=True)  # Сохранять очередь
    channel.basic_qos(prefetch_count=1)

    threads = []
    on_message_callback = functools.partial(on_message, args=(
        connection, threads, do_function, initArgs))

    channel.basic_consume(queue_name,
                          on_message_callback)
    # logger = setup_logger(initArgs, initArgs)
    # logger.info('Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()  # Потребление сообщений. Отсюда никогда не выйти. Только убив скрипт
    for thread in threads:
        thread.join()
    connection.close()


@retry(exceptions=(
        ConnectionWrongStateError, StreamLostError, ChannelClosedByBroker, AMQPConnectionError,
        AMQPConnectorStackTimeout),
    tries=-1, delay=300)
def send(data, queue_name, user, password, logger):
    credentials = pika.PlainCredentials(user, password)

    parameters = pika.ConnectionParameters(broker_ip,
                                           broker_port,
                                           '/',
                                           credentials)
    connection = pika.BlockingConnection(parameters)

    channel = connection.channel()

    channel.queue_declare(queue=queue_name,  # Название очереди
                          durable=True)  # Сохранять очередь

    channel.basic_publish(exchange='',
                          routing_key=queue_name,  # Название очереди
                          body=data,
                          properties=pika.BasicProperties(
                              delivery_mode=2  # Сохранять сообщение
                          ))
    logger.debug("Sent data " + data)
    connection.close()
