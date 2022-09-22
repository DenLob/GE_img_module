import json

from sqlalchemy import desc

from database.models import create_model


def insert_plant(Plant, TypePlant, session, center, pallet_id, type_plant=None):
    if type_plant is None:
        type_plant = 'Undefined'
    center = json.dumps({'x': center[0], 'y': center[1], 'x_w': center[2], 'y_h': center[3]})
    last_id = session.query(Plant).order_by(desc(Plant.id)).first()
    id_type_plants = session.query(TypePlant).filter(TypePlant.name == type_plant).one()
    id_type_plants = id_type_plants.id
    if last_id is None:
        new_id = 1
    else:
        new_id = last_id.id + 1
    new_plant = Plant(
        id=new_id,
        center=center,
        id_type_plants=id_type_plants,
        id_pallet=pallet_id
    )
    session.add(new_plant)


def insert_pallet_and_plant(pallet_data, session):
    (Pallet, Plant, TypePlant) = create_model()
    last_id = session.query(Pallet).order_by(desc(Pallet.id)).first()
    has_plants = True
    if pallet_data['type_plant'] == '':
        has_plants = False
    if pallet_data['type_plant'] is None:
        type_plant = 'Undefined'
    else:
        type_plant = pallet_data['type_plant']
    if has_plants:
        id_type_plant = session.query(TypePlant).filter(TypePlant.name == type_plant).one()
        id_type_plant = id_type_plant.id
    if last_id is None:
        new_id = 1
    else:
        new_id = last_id.id + 1
    if has_plants:
        new_pallet = Pallet(
            id=new_id,
            time_1=pallet_data['time'],
            path_1=pallet_data['path'],
            id_type_plant=id_type_plant
        )
    else:
        new_pallet = Pallet(
            id=new_id,
            time_1=pallet_data['time'],
            path_1=pallet_data['path']
        )
    session.add(new_pallet)
    session.query(Plant).filter(Plant.id_pallet == new_id).delete(synchronize_session='fetch')
    for i in range(0, len(pallet_data['plants'])):
        center = pallet_data['plants'][i]['center']
        type_plant = pallet_data['plants'][i]['type_plant']
        insert_plant(center=center, pallet_id=new_id, type_plant=type_plant, session=session, Plant=Plant,
                     TypePlant=TypePlant)


def update_pallet(data, Session):
    session = Session()
    for key in data:
        pallet_id = data[key]['id']
        if pallet_id is None:
            insert_pallet_and_plant(data[key], session)
        else:
            (Pallet, Plant, TypePlant) = create_model()
            new_pallet = session.query(Pallet).get(pallet_id)
            has_plants = True
            if data[key]['type_plant'] == '':
                has_plants = False
            if data[key]['type_plant'] is None:
                type_plant = 'Undefined'
            else:
                type_plant = data[key]['type_plant']
            if has_plants:
                id_type_plant = session.query(TypePlant).filter(TypePlant.name == type_plant).one()
                id_type_plant = id_type_plant.id
                new_pallet.id_type_plant = id_type_plant

            new_pallet.time_1 = data[key]['time']
            new_pallet.path_1 = data[key]['path']

            new_pallet.time_3 = None
            new_pallet.path_2 = None
            new_pallet.path_3 = None
            new_pallet.id_line = None
            session.add(new_pallet)
            # session.commit()
            session.query(Plant).filter(Plant.id_pallet == pallet_id).delete(synchronize_session='fetch')
            # session.commit()
            for i in range(0, len(data[key]['plants'])):
                center = data[key]['plants'][i]['center']
                type_plant = data[key]['plants'][i]['type_plant']
                insert_plant(center=center, pallet_id=pallet_id, type_plant=type_plant, session=session, Plant=Plant,
                             TypePlant=TypePlant)
        session.commit()
        session.close()
