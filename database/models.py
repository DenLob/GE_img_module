from time import sleep

from retry import retry
from sqlalchemy import Table, MetaData, create_engine
from sqlalchemy.exc import OperationalError, StatementError
from sqlalchemy.orm.query import Query as _Query
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from constants import db_host
from log_dir.loggers import db_logger

# offset = datetime.timedelta(hours=3)
# tz = datetime.timezone(offset, name='МСК')

user = "***********"
password = "*********"
host = db_host
database = "db_system"
engine = create_engine('postgresql+psycopg2://{0}:{1}@{2}/{3}'.format(user, password, host, database), pool_size=10,
                       max_overflow=2,
                       pool_recycle=300,
                       pool_pre_ping=True,
                       pool_use_lifo=True)


class RetryingQuery(_Query):
    __max_retry_count__ = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        attempts = 0
        while True:
            attempts += 1
            try:
                return super().__iter__()
            except OperationalError as ex:
                if "server closed the connection unexpectedly" not in str(ex):
                    raise
                if attempts <= self.__max_retry_count__:
                    sleep_for = 2 ** (attempts - 1)
                    db_logger.error(
                        "/!\ Database connection error: retrying Strategy => sleeping for {}s"
                        " and will retry (attempt #{} of {}) \n Detailed query impacted: {}".format(
                            sleep_for, attempts, self.__max_retry_count__, ex)
                    )
                    sleep(sleep_for)
                    continue
                else:
                    raise
            except StatementError as ex:
                if "reconnect until invalid transaction is rolled back" not in str(ex):
                    raise
                self.session.rollback()


@retry(exceptions=(
        OperationalError,),
    tries=-1, delay=300)
def create_model():
    Base = declarative_base(bind=engine)
    metadata = MetaData(bind=engine)

    class Pallet(Base):
        """"""
        __table__ = Table('pallet', metadata, autoload=True)
        __tablename__ = 'pallet'
        __table_args__ = {'autoload': True}

    class Plant(Base):
        """"""
        __table__ = Table('plant', metadata, autoload=True)
        __tablename__ = 'plant'
        __table_args__ = {'autoload': True}

    class TypePlant(Base):
        """"""
        __table__ = Table('type_plant', metadata, autoload=True)
        __tablename__ = 'type_plant'
        __table_args__ = {'autoload': True}

    Base.metadata.create_all(engine)
    return (Pallet, Plant, TypePlant)


def create_sessionmaker():
    Session = sessionmaker(bind=engine, query_cls=RetryingQuery)
    return Session
