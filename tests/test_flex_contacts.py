"""Native flex cables must work with the package's public contact queries."""

import mujoco as mj
import numpy as np

from mjsim.utils.mj import (
    ObjType,
    get_bodies_in_contact,
    get_contact_states,
    get_geoms_in_contact,
    get_names,
    get_number_of,
    id2name,
)
from mjsim.utils.mjs import cable, empty_scene


def test_queries_distinguish_geom_and_flex_contacts():
    spec = empty_scene(integrator="discrete", solver="CG", noslip_iterations=0)
    spec.geom("floor").surfacevel = [0.2, 0, 0, 0, 0, 0]
    rigid = spec.worldbody.add_body(name="rigid_body", pos=[0.5, 0, 0.03])
    rigid.add_freejoint()
    rigid.add_geom(name="rigid_geom", size=[0.05], mass=1)
    rigid.add_site(name="probe")
    spec.add_sensor(
        name="position",
        type=mj.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mj.mjtObj.mjOBJ_SITE,
        objname="probe",
    )
    spec.attach(
        cable(model_name="rod", n_segments=4, length=0.2, segment_size=0.01),
        prefix="cable/",
        frame=spec.worldbody.add_frame(pos=[0, 0, 0.005]),
    )
    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert data.ncon > 1
    assert get_number_of(model, ObjType.SENSOR) == 1
    assert get_names(model, ObjType.SENSOR) == ["position"]
    assert get_number_of(model, ObjType.FLEX) == 1
    assert id2name(model, 0, ObjType.FLEX) == "cable/rod"
    pairs = get_geoms_in_contact(model, data)
    assert pairs and all(set(pair) == {"floor", "rigid_geom"} for pair in pairs)
    pairs = get_bodies_in_contact(model, data)
    assert pairs and all(set(pair) == {"world", "rigid_body"} for pair in pairs)
    found, contacts = get_contact_states(data, model)
    assert found and len(contacts) == data.ncon
    flex_contacts = [contact for contact in contacts if np.any(contact.flex >= 0)]
    assert flex_contacts
    for contact in flex_contacts:
        assert None in contact.geom_names
        assert "floor" in contact.geom_names
        np.testing.assert_allclose(contact.normal, [0, 0, 1])
        assert np.isfinite(contact.w_wrench).all()
    rigid_contacts = [contact for contact in contacts if np.all(contact.geom >= 0)]
    assert rigid_contacts
    assert np.linalg.norm(rigid_contacts[0].surfacevel) > 0
    for contact in contacts:
        np.testing.assert_array_equal(
            contact.surfacevel, data.contact[contact.index].surfacevel
        )
