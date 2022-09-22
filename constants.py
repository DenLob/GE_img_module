import numpy as np
import http.client

CONST_IP = '***********8'
conn = http.client.HTTPConnection("ifconfig.me")
conn.request("GET", "/ip")
if conn.getresponse().read().decode('utf-8') == CONST_IP:
    CONST_RESULT_FOLDER = '/root/IMG_stitch/result_photos/'
    CONST_MEDIA_FOLDER = '/root/GE_web/ge_web/ge_server/media/'
    broker_ip = 'localhost'         # IP-адресс брокера.
    db_host = 'localhost'           # IP-адерс БД теплицы
else:
    CONST_RESULT_FOLDER = 'D:/GreenStuff/GE_image_stitching/result_photos/'
    CONST_MEDIA_FOLDER = 'D:/GreenStuff/GE_web/server/ge_server/media/'
    broker_ip = '***********'    # IP-адресс брокера. Совпадает с IP-адресом главного сервера
    db_host = '**************'      # IP-адерс БД теплицы

CONST_RAW_FOLDER = 'D:/GreenStuff/GE_image_stitching/photos/tmp_1/'

CONST_CAMERA_MATRIX = np.array([[1.05978156e+04, 0.00000000e+00, 1.05151495e+03],
                                [0.00000000e+00, 9.74153052e+03, 7.55872360e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
CONST_DIST_COEFS = np.array([-1.82813831e+01, 7.80658015e+02, 1.43203165e-03, -3.72716007e-02, -2.13825863e+04])

broker_port = 5672  # Порт, по которому происходит общение брокера
