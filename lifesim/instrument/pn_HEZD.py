"""Optically thin, single-size graphite HEZD for LIFEsim (Python >=3.8).

Gra_81 supplies temperature-independent Q_abs; scattered light is not included.
noise(index) returns photons/s/bin. signal(index) retains the original six-item
interface.
"""
from pathlib import Path
from functools import lru_cache
import re
import warnings
import numpy as np
from lifesim.core.modules import PhotonNoiseUniverseModule, TransmissionModule
from lifesim.util.constants import h, c, k, radius_sun as rsun, m_per_au as au

# SI constants absent from lifesim.util.constants (could also be implented into lifesim.util.constants)
sb = 5.670374419e-8     # Stefan-Boltzman constant [W m^-2 K^-4]
lsun = 3.828e26         # Luminosty Sun [W]
mearth = 5.9722e24      # Earth Mass [kg]
integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# Reference defaults Tau Ceti.
reference_temp_s = 5283.0           # [K]
reference_l_sun = 0.4672            # solar luminosities
reference_radius_s = np.sqrt(reference_l_sun * lsun /(4 * np.pi * sb * reference_temp_s**4)) / rsun
reference_grain_size_um = 0.1       # grain radius [um]
reference_dust_mass = 5.98e-10      # total mass for hezd=1 in earth masses
reference_density = 2240.0          # [kg/m^3]
reference_t_sub = 2000.0            # [K]
reference_b = 11.5                  # R_out = b * R_in
reference_alpha = 0                 # surface density profile parameter


def resolve_stellar_parameters(temp_s=None, radius_s=None, l_sun=None):
    """
    Determine consistent stellar parameters from two supplied quantities.
    
    Parameters
    ----------
    temp_s : float or None, optional
        Effective temperature of the star in [K].
    radius_s : float or None, optional
        Radius of the star in [solar radii].
    l_sun : float or None, optional
        Luminosity of the star in [solar luminosities].
    
    Returns
    -------
    temp_s : float
        Effective stellar temperature in [K].
    radius_s : float
        Stellar radius in [solar radii].
    l_sun : float
        Stellar luminosity in [solar luminosities].
    
    Raises
    ------
    ValueError
        If exactly one of the three stellar parameters is supplied.
    
    Notes
    -----
    The quantities are related by the Stefan-Boltzmann law.
    If no parameters are supplied, the reference Tau Ceti model is used.
    If all three parameters are given, temperature and radius take priority;
    the supplied luminosity is ignored and a warning is issued.
    """
    count = sum(v is not None for v in (temp_s, radius_s, l_sun))
    if count == 0:
        temp_s, radius_s = reference_temp_s, reference_radius_s
    elif count == 1:
        raise ValueError('Supply two stellar parameters: temp_s, radius_s, l_sun.')
    coefficient = 4 * np.pi * rsun**2 * sb / lsun
    if temp_s is not None and radius_s is not None:
        if count == 3:
            warnings.warn('All three stellar parameters supplied: using temp_s and '
                          'radius_s; supplied l_sun is ignored.', UserWarning, stacklevel=2)
        l_sun = coefficient * float(radius_s)**2 * float(temp_s)**4
    elif radius_s is None:
        radius_s = np.sqrt(float(l_sun) / (coefficient * float(temp_s)**4))
    else:
        temp_s = (float(l_sun) / (coefficient * float(radius_s)**2))**0.25
    return float(temp_s), float(radius_s), float(l_sun)


@lru_cache(maxsize=4)
def _read_gra81(path):
    """
    Read, sort, and cache the graphite absorption-efficiency table Gra_81.

    Parameters
    ----------
    path : str
        Path to the Gra_81 optical-property file.

    Returns
    -------
    radii : np.ndarray
        Available grain radii in [micrometers], sorted in ascending order.
    wavelength : np.ndarray
        Wavelength grid in [m], sorted in ascending order.
    values : np.ndarray
        Dimensionless absorption efficiencies Q_abs for each grain radius and
        wavelength; shape (n_radii, n_wavelengths).

    Raises
    ------
    ValueError
        If the file contains no complete grain-radius blocks or if its blocks
        do not share the same wavelength grid.

    Notes
    -----
    The file's data are cached by path to avoid reading and parsing it repeatedly.
    """
    radii, blocks, waves, q = [], [], [], []
    reference = None
    def finish():
        """
        Validate and store the current grain-radius wavelength block.
        
        Returns
        -------
        None
            Updates the enclosing wavelength-grid reference and the list of
            absorption-efficiency blocks.
        
        Raises
        ------
        ValueError
            If the current block has a different wavelength grid from the first
            completed block.
        
        Notes
        -----
        This helper uses the enclosing ``waves``, ``q``, ``reference``, and
        ``blocks`` variables. Empty wavelength blocks are ignored.
        """
        nonlocal reference
        if not waves:
            return
        w = np.asarray(waves)
        if reference is None:
            reference = w
        elif w.shape != reference.shape or not np.allclose(w, reference):
            raise ValueError('Gra_81 blocks must share the wavelength grid.')
        blocks.append(q.copy())
    with open(path, encoding='ascii', errors='replace') as stream:
        for line in stream:
            match = re.match(r'\s*([+\-0-9.EeDd]+)\s*=\s*radius\(micron\)', line, re.I)
            if match:
                finish()
                radii.append(float(match.group(1).replace('D', 'E').replace('d', 'e')))
                waves, q = [], []
            elif radii:
                fields = line.replace('D', 'E').replace('d', 'e').split()
                try:
                    w, v = float(fields[0]), float(fields[1])
                except (ValueError, IndexError):
                    continue
                waves.append(w)
                q.append(v)
    finish()
    if not blocks or len(blocks) != len(radii):
        raise ValueError('No complete Gra_81 radius blocks found.')
    ri, wi = np.argsort(radii), np.argsort(reference)
    radii, wavelength = np.asarray(radii)[ri], reference[wi] * 1e-6
    values = np.asarray(blocks)[ri][:, wi]
    return radii, wavelength, values


def _planck(wavelength, temperature):
    """
    Calculate the blackbody energy spectral radiance from Planck's law.
    
    Parameters
    ----------
    wavelength : float or np.ndarray
        Wavelength or wavelengths in [m].
    temperature : float or np.ndarray
        Blackbody temperature or temperature map in [K].
    
    Returns
    -------
    B_lambda : float or np.ndarray
        Energy spectral radiance in [W m^-2 sr^-1 m^-1]. The output follows
        NumPy broadcasting of the wavelength and temperature inputs.
    
    Notes
    -----
    For positive wavelengths, zero temperature produces zero radiance.
    ``np.expm1`` improves numerical accuracy at small exponents, while
    exponents larger than 700 are treated as negligible emission to avoid
    floating-point overflow. This function returns energy, not photon,
    spectral radiance. Could also be implemented/changed in lifesim.util.radiation
    """
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        x = h * c / (np.asarray(wavelength) * k * np.asarray(temperature))
        return np.where(x > 700, 0.0,
                        2 * h * c**2 / np.asarray(wavelength)**5 / np.expm1(np.minimum(x, 700)))


class PhotonNoiseHEZD(PhotonNoiseUniverseModule):
    """
    Simulate thermal photon leakage from optically thin, single-size graphite HEZD.

    Notes
    -----
    Model parameters are initialized from the reference values defined above and
    can be adjusted on the object after construction, as in PhotonNoiseExozodi.
    For hezd=1, the dust mass equals ``self.dust_mass`` in Earth masses.
    ``r_in=None`` uses the grain sublimation radius; ``r_out=None`` uses
    ``self.b * r_in``. Inclination and PA are specified in degrees.

    TM3 transmission is evaluated by the connected LIFEsim TransmissionMap
    module at the HEZD pixel positions, preserving the fine dust-ring grid.
    """

    # Used by Instrument.get_snr to scale a unit-level reference spectrum.
    noise_level_column = 'hezd'
    noise_level_default = 1.0  # Keep legacy behavior for catalogs without HEZD.

    def __init__(self, name: str):
        """
        Initialize the HEZD module with its reference model parameters.

        Parameters
        ----------
        name : str
            Unique name of this module on the LIFEsim bus.

        Notes
        -----
        Model parameters may be changed through ``self.grain_size_um``,
        ``self.dust_mass``, ``self.alpha``, ``self.b``, ``self.r_in``,
        ``self.r_out``, ``self.inc``, ``self.PA``, and other attributes
        before calling ``noise`` or ``signal``.
        """
        super().__init__(name=name)
        self.add_socket(s_name='transmission_hezd', s_type=TransmissionModule)

        # HEZD reference model parameters
        self.gra_file = None
        self.grain_size_um = reference_grain_size_um
        self.dust_mass = reference_dust_mass
        self.alpha = reference_alpha
        self.r_in = None
        self.r_out = None
        self.b = reference_b
        self.grain_density_kg_m3 = reference_density
        self.t_sub_k = reference_t_sub

        # Disk orientation
        self.inc = 0.0
        self.PA = 0.0

        # Cache for grain radiative-equilibrium calculations
        self._thermal_cache = None

    def _optics(self):
        """Load and interpolate graphite absorption efficiencies for the grain size.
        
        Returns
        -------
        wavelength : np.ndarray
            Wavelength grid of the Gra_81 table in [m].
        q : np.ndarray
            Dimensionless absorption-efficiency spectrum Q_abs at the configured
            grain radius.
        key : tuple
            Cache-identification tuple containing the resolved file path and
            grain radius in [micrometers].
        
        Raises
        ------
        FileNotFoundError
            If the Gra_81 table cannot be found.
        ValueError
            If ``grain_size_um`` lies outside the grain-radius range in the table,
            or if the underlying table is invalid.
        
        Notes
        -----
        Q_abs is interpolated linearly between table values in log10(grain radius),
        not by selecting the nearest grain size. No extrapolation beyond the table
        is performed. Table data are cached using the resolved file path.
        """
        path = Path(self.gra_file).expanduser() if self.gra_file is not None else Path(__file__).with_name('Gra_81')
        if self.gra_file is None:
            path = Path(__file__).resolve().parent.parent / 'data' / 'Gra_81'
        else:
            path = Path(self.gra_file).expanduser()
            
        if not path.is_file():
            raise FileNotFoundError(f'Gra_81 file not found: {path}')
        path = path.resolve()
        radii, wavelength, table = _read_gra81(str(path))
        a = self.grain_size_um

        if a < radii[0] or a > radii[-1]:
            raise ValueError('grain_size_um={} outside Gra_81 range [{}, {}] um.'.format(a, radii[0], radii[-1]))
        q = np.array([np.interp(np.log10(a), np.log10(radii), col) for col in table.T])
        return wavelength, q, (str(path), a)

    def equilibrium(self, temp_s=None, radius_s=None, l_sun=None):
        """
        Solve the dust-grain radiative equilibrium and sublimation radius.
        
        Parameters
        ----------
        temp_s : float or None, optional
            Effective temperature of the host star in [K].
        radius_s : float or None, optional
            Radius of the host star in [solar radii].
        l_sun : float or None, optional
            Luminosity of the host star in [solar luminosities].
        
        Returns
        -------
        rsub : float
            Grain sublimation radius at ``self.t_sub_k`` in [AU].
        temperature : callable
            Function accepting a distance from the star in [AU] and returning
            the radiative-equilibrium grain temperature in [K]. Supports arrays.
        
        Notes
        -----
        The stellar parameters follow ``resolve_stellar_parameters``: supply any
        two, or none to use the reference model. The heating and cooling integrals
        use Q_abs over the full Gra_81 wavelength grid. The temperature lookup
        is cached for the optical data, stellar parameters, and sublimation
        temperature; it does not depend on the LIFEsim instrument wavelength bins.
        The calculation assumes thermal absorption and emission without scattering.
        """
        temp_s, radius_s, _ = resolve_stellar_parameters(temp_s, radius_s, l_sun)
        wavelength, q, key = self._optics()
        t_sub = self.t_sub_k
        key += (temp_s, radius_s, t_sub)
        if self._thermal_cache is None or self._thermal_cache[0] != key:
            # Temperature grid for radiative-equilibrium inversion.
            temperatures = np.unique(np.r_[np.geomspace(1.0, 1e5, 12000), t_sub])
            emission = integrate(q * _planck(wavelength[None, :], temperatures[:, None]), wavelength, axis=1)
            stellar = integrate(q * _planck(wavelength, temp_s), wavelength)
            sub_emission = integrate(q * _planck(wavelength, t_sub), wavelength)
            rsub = radius_s * rsun / (2 * au) * np.sqrt(stellar / sub_emission)
            self._thermal_cache = (key, temperatures, emission, stellar, rsub)
        _, temperatures, emission, stellar, rsub = self._thermal_cache
        def temperature(radius_au):
            """
            Interpolate the equilibrium dust temperature at given radii.
            
            Parameters
            ----------
            radius_au : float or np.ndarray
                Radial distance or distances from the host star in [AU].
            
            Returns
            -------
            temperature_k : float or np.ndarray
                Interpolated grain equilibrium temperature in [K], matching the
                input shape.
            
            Notes
            -----
            Radiative equilibrium is inverted using the precomputed emission table.
            Requested emission values outside the lookup range are clamped to its
            endpoint temperatures.
            """
            r = np.asarray(radius_au, dtype=float)
            required = (radius_s * rsun / (r * au))**2 * stellar / 4
            return np.interp(required, emission, temperatures)
        return rsub, temperature

    def _calculate(self, index, level=None):
        """
        Calculate projected HEZD photon maps before and after transmission.
        
        Parameters
        ----------
        index : int or None
            Index of the planet in ``data.catalog``. If None, use stellar and dust
            parameters from ``data.single`` instead.
        level : float or None, optional
            Override the system HEZD level for this calculation. If None,
            use the `hezd` value from the data (defaulting to 1.0).
        
        Returns
        -------
        sky_r : np.ndarray
            Projected radial distance from the star in [AU], broadcast to the
            three-dimensional wavelength-by-image grid.
        sigma : np.ndarray
            Dimensionless projected grain cross-sectional area per sky area,
            including the system's HEZD level.
        temp_map : np.ndarray
            Grain temperature in [K] at each image position.
        disk : np.ndarray
            HEZD photon rate before TM3 transmission in [photons s^-1 bin^-1 pixel^-1].
        transmitted : np.ndarray
            HEZD photon rate after TM3 transmission in
            [photons s^-1 bin^-1 pixel^-1].
        sigma_0 : float
            Dimensionless reference surface-density normalization set by the
            reference dust mass, excluding the system's ``hezd`` level.
        
        Notes
        -----
        The first five returned arrays have shape
        (n_wavelengths, image_size, image_size). A pixel grid spanning the disk
        is projected using the inclination and position angle. The map is then
        masked to the instrument's wavelength-dependent field of view. The
        TM3 transmission is evaluated by the connected TransmissionMap module
        at the actual HEZD pixel positions (``direct_mode=True``) rather than
        interpolating ``data.inst['t_map']``.
        The input includes stellar properties (``temp_s``, ``radius_s``, or a
        suitable pair with ``l_sun``), ``distance_s`` in [pc], and the system's
        ``hezd`` level; when absent, distance defaults to 3.65 pc and HEZD level
        to 1. Both photon maps include collecting area and field-of-view masking,
        but exclude integration time, overall throughput, and detector efficiency.
        ``self.rad_pix`` and ``self.mas_pix`` record the angular pixel scale.
        """
        row = self.data.single if index is None else self.data.catalog.iloc[index]
        t, r = row.get('temp_s'), row.get('radius_s')
        lum = row.get('l_sun') if t is None or r is None else None
        rsub, temperature = self.equilibrium(t, r, lum)
        distance = row.get('distance_s', 3.65)
        hezd_level = row.get('hezd', 1.0) if level is None else level
        rin = rsub if self.r_in is None else self.r_in
        rout = self.b * rin if self.r_out is None else self.r_out
        self.r_sub_au, self.r_in_actual, self.r_out_actual = rsub, rin, rout

        # Normalize geometric cross section to the chosen total mass.
        exponent = 2 - self.alpha
        log_ratio = np.log(rout / rin)
        integral = (log_ratio if exponent == 0 else
                    np.expm1(exponent * log_ratio) / exponent)
        cross_section = (3 * self.dust_mass * mearth /
                         (4 * self.grain_density_kg_m3 * self.grain_size_um * 1e-6))
        self.sigma_0 = cross_section / (2 * np.pi * (rin * au)**2 * integral)

        inst = self.data.inst
        wl = np.asarray(inst['wl_bins'])[:, None, None]
        widths = np.asarray(inst['wl_bin_widths'])[:, None, None]
        wave, q, _ = self._optics()
        qbin = np.interp(wl, wave, q)

        # Pixel centers span only the ring. Evaluate transmission at those
        # exact positions; retain the instrument's physical field-of-view cut.
        n = self.data.options.other['image_size']
        au_pix = 2 * rout / n
        coordinates = (np.arange(n) - (n - 1) / 2) * au_pix
        x, y = np.meshgrid(coordinates, coordinates)
        rad_per_au = np.pi / (180 * 3600) / distance
        self.rad_pix = au_pix * rad_per_au
        self.mas_pix = au_pix / distance * 1000
        angle, cosine = np.deg2rad(self.PA), np.cos(np.deg2rad(self.inc))
        xp = x * np.cos(angle) + y * np.sin(angle)
        yp = -x * np.sin(angle) + y * np.cos(angle)
        disk_r = np.sqrt(xp**2 + (yp / cosine)**2)
        inside = (disk_r >= rin) & (disk_r <= rout)
        temp_map, sigma = np.zeros_like(disk_r), np.zeros_like(disk_r)
        temp_map[inside] = temperature(disk_r[inside])
        sigma[inside] = (self.sigma_0 * hezd_level *
                         (disk_r[inside] / rin)**(-self.alpha) / cosine)
        sky_r = np.hypot(x, y)
        aperture = (sky_r * rad_per_au <=
                    np.asarray(inst['hfov'])[:, None, None])
        # Evaluate TM3 through LIFEsim's transmission module on the HEZD grid.
        transmission = self.run_socket(
            s_name='transmission_hezd', method='transmission_map',
            map_selection=['tm3'], direct_mode=True,
            d_alpha=x * rad_per_au, d_beta=y * rad_per_au)[2]

        # B_lambda -> photons/s/bin/pixel before instrument throughput.
        photons = _planck(wl, temp_map) * widths * wl / (h * c)
        disk = (photons * qbin * sigma * self.rad_pix**2 *
                inst['telescope_area'] * aperture)
        transmitted = disk * transmission
        shape = disk.shape
        return (np.broadcast_to(sky_r, shape), np.broadcast_to(sigma, shape),
                np.broadcast_to(temp_map, shape), disk, transmitted, self.sigma_0)

    def signal(self, index=None):
        """
        Return HEZD spatial maps for inspection and plotting.
        
        Parameters
        ----------
        index : int or None, optional
            Index of the planet in ``data.catalog``. If None, use the parameters
            in ``data.single``.
        
        Returns
        -------
        sky_r : np.ndarray
            Projected radial distance from the star in [AU].
        sigma : np.ndarray
            Projected, dimensionless dust cross-sectional density.
        temp_map : np.ndarray
            Projected dust temperature in [K].
        disk : np.ndarray
            Photon rate before TM3 transmission in [photons s^-1 bin^-1 pixel^-1].
        transmitted : np.ndarray
            Photon rate after TM3 transmission in [photons s^-1 bin^-1 pixel^-1].
        sigma_0 : float
            Reference dust surface-density normalization, excluding system
            HEZD level.
        
        Notes
        -----
        The returned maps have shape (n_wavelengths, image_size, image_size)
        and are identical to those from ``_calculate``. The transmitted map is a
        single-output TM3 map, not the phase-chopped TM3-minus-TM4 signal. Use
        ``self.mas_pix`` when displaying the maps in milliarcseconds.
        """
        return self._calculate(index)

    def noise(self, index=None, level=None):
        """
        Simulate the HEZD photon rate leaking into the LIFE array measurement.
        
        Parameters
        ----------
        index : int or None, optional
            Specifies the planet for which to calculate the HEZD contribution.
            If an integer n is given, the data are taken from the n-th row of
            ``data.catalog``. If None, the parameters in ``data.single`` are used.
        level : float or None, optional
            Override the HEZD level. If None, read `hezd` from the current
            system, defaulting to 1.0. Use 1.0 for the reference spectrum.

        Returns
        -------
        hezd_leak : np.ndarray
            HEZD leakage rate in [photons s^-1] per wavelength bin.
        
        Notes
        -----
        The model uses the stellar parameters ``temp_s``, ``radius_s`` and,
        when needed, ``l_sun``; ``distance_s`` in [pc]; and the dimensionless
        ``hezd`` level, where 1 corresponds to ``self.dust_mass``.
        Disk parameters, grain size, and orientation are stored in this module.
        The calculation uses wavelength-bin centers and widths from ``data.inst``,
        the combined aperture area in [m^2], the
        configured ``image_size``, the instrument field of view, and TM3
        transmission evaluated through the connected TransmissionMap module.
        
        The returned quantity is the sum of transmitted photon rates over all
        dust pixels, not the square-root shot-noise uncertainty. It includes
        TM3 transmission, field-of-view masking and collecting area, but not
        integration time, overall throughput or detector efficiency.
        """
        return self._calculate(index, level=level)[4].sum(axis=(-2, -1))
