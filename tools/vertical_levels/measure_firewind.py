import sys; sys.path.insert(0,'src'); sys.path.insert(0,'tests')
import numpy as np, torch
import physics_constants as pc
from conftest import make_config
from simulation import Simulation
from scenarios import make_config as mc
from gen_maps import MapGenerator
from simulation import SimState
G=64
def run():
    cfg=make_config(G, fire=True, radiation=True, relaxation=True)
    cfg.setdefault("units",{})["seconds_per_tick"]=1.0
    sim=Simulation(cfg)
    gen=MapGenerator(cfg); terr=np.zeros((G,G))
    air=gen.air_temperature_profile(terr); mass=gen.boundary_layer_mass(terr); oxy=gen.oxygen_profile(terr,air)
    t=lambda a: torch.as_tensor(a,dtype=sim.dtype)
    fuel=torch.full((1,G,G),1.0,dtype=sim.dtype); ft=torch.full((1,G,G),pc.T_REF,dtype=sim.dtype)
    ft[0,G//2,G//2]=1200.0
    z=torch.zeros_like(t(mass))
    s=SimState(mass=t(mass),air_temperatures=t(air).unsqueeze(-3),ground_temperature=t(air).clone(),
        fuel_temperatures=ft,fuel=fuel,oxygen=t(oxy),terrain=t(terr),
        temp_eq=t(air).clone().unsqueeze(-3),oxygen_eq=t(oxy).clone(),mass_eq=t(mass).clone(),
        x_wind_vel=z.clone(),y_wind_vel=z.clone(),u_amb_x=torch.zeros((1,1),dtype=sim.dtype),u_amb_y=torch.zeros((1,1),dtype=sim.dtype),
        radiant_flux=z.clone())
    peak=0.0
    for _ in range(120):
        sim.step_fields(s)
        peak=max(peak,float(torch.sqrt(s.x_wind_vel**2+s.y_wind_vel**2).max()))
    return peak
print(f"peak fire-driven wind = {run():.4e} m/s")
