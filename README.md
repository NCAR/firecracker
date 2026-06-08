# Firecracker
## Description
This project implements a simple fire simulation as a 2D grid of cells using a combustion model for fire behavior.

## Diffusion and Advection Physics
### Diffusion
Every cell has a temperature associated with it that is used for fire ignition and wind. Every simulation step, heat from hotter cells diffuses into cooler cells by applying a 3x3 Gaussian convolution to all cells multiple times, which causes the temperature of each cell to become a weighted average of all nearby cells with closer cells contributing more than distant cells. This results in a blurring effect where sharp temperature differences between neighboring cells are smoothed out. After many steps, without more heat being continually added or removed from the system, the system reaches maximum entropy and all cells converge to the same temperature.
### Advection
Since warm air rises, cells with higher temperatures than their neighbors will have a lower air pressure at their surface than surrounding cells, causing cooler air to flow in to equalize the pressure. This results in air moving from cooler regions to warmer regions (high pressure to low pressure), or wind. The rate at which air flows depends on the pressure difference between neighboring cells, with higher temperature gradients often resulting in higher pressure gradients and thus faster/stronger winds. Wind also carries heat with it, meaning that as air flows from cooler regions to warmer regions, it cools down cells that it flows through and the moving air warms up in the process. Because temperature gradients impact pressure gradients which cause wind, and because wind carries heat (thus changing the temperature gradient and thus also the pressure gradient), the wind direction and magnitude at each cell is constantly changing as the system evolves.
Like with diffusion, advection is computed by applying a 3x3 convolution to all cells, however unlike diffusion, the kernel is not symmetric and more weight is put on neighboring cells in the opposite direction of the cell's wind vector. The weight put on neighboring cells depends both on the wind direction and its magnitude.

## Fire Behavior
Fire needs three things to exist: heat, fuel, and oxygen. The prescence of all three ingredients will result in fire, and the absence of at least one of them will result in no fire.
### Heat
Fire will only ignite once the cell reaches a certain theshold for fuel temperature (which is separate from air temperature). When this happens, the fire will begin generating more heat proportional to the amount of fuel and oxygen being consumed (some of which will go back into the fuel, and some into the air). If the fuel temperature drops below the ignition point (which can happen if nearby cells are very cool and the wind is very strong for long periods of time), then the fire will extinguish. The fuel temperature is updated using the difference between the fuel temperature and the air temperature. Every simulation step, some constant percentage of the fuel-air temperature difference is used to compute how much heat from the air will be transferred to the fuel or vise-versa.
### Fuel
Once a fire ignites, it will begin consuming fuel at a rate proportional to the cell's fuel temperature. If the fuel remaining for a cell reaches zero (or an amount very close to it), then the fire will extinguish.
### Oxygen
Once a fire ignites, it will begin consuming oxygen at a rate proportional to the cell's fuel temperature. If the oxygen remaining for a cell reaches zero (or an amount very close to it), then the fire will extinguish. Note: if there is still fuel and the cell's fuel temperature is still above the ignition point when oxygen returns, then the fire will re-ignite.
