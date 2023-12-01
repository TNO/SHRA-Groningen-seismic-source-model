import tempfile
import os
import yaml
import xarray as xr
from unittest import TestCase
from calibrate_ssm import main as main_calibrate
from forecast_ssm import main as main_forecast
from chaintools.chaintools import tools_configuration as cfg


class IntegrationTests(TestCase):
    test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    def test_ssm_calibrate(self):
        """
        Integration test of seismic source model calibration. Asserts if the calibration workflow runs without errors.
        Asserts correct dimensions and coordinates of SSM calibration.

        Returns
        -------

        """

        # arrange
        # -------
        module = 'calibrate_ssm'
        data_sink = 'calibration_data'
        test_config_yaml = os.path.join(self.test_data_path, 'test_configs/test_config_ssm.yml')
        module_config = cfg.configure(test_config_yaml, module)
        test_group = module_config['data_sinks'][data_sink]['group']
        calibration_expected_file = os.path.join(self.test_data_path, 'res/ssm_calibration_test.h5')
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            test_outcome_file = [test_out_dir, 'ssm_calibration_out.h5']
            module_config['data_sinks'][data_sink]['path'] = test_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            main_calibrate(temp_config_path)

            # assert
            # ------
            # assert correct dimensions, coordinates, and data values
            outcome_path = os.path.join(test_outcome_file[0], test_outcome_file[1])
            activity_rate_test = self.load_from_file(outcome_path, group=f"{test_group}/activity_rate_model")
            magnitudes_test = self.load_from_file(outcome_path, group=f"{test_group}/magnitude_model")
            stress_test = self.load_from_file(outcome_path, group=f"{test_group}/dsm_pmf")

            activity_rate_expected = self.load_from_file(calibration_expected_file, group="calibration/activity_rate_model")
            magnitudes_expected = self.load_from_file(calibration_expected_file, group="calibration/magnitude_model")
            stress_expected = self.load_from_file(calibration_expected_file, group="calibration/dsm_pmf")

            xr.testing.assert_allclose(stress_test, stress_expected)
            xr.testing.assert_allclose(activity_rate_test, activity_rate_expected)
            xr.testing.assert_allclose(magnitudes_test, magnitudes_expected)

    def test_ssm_forecast(self):
        """
        Integration test of seismic source model forecasting. Asserts if the forecasting workflow runs without errors.
        Asserts correct dimensions and coordinates of SSM forecast.

        Returns
        -------

        """

        # arrange
        # -------
        module = 'forecast_ssm'
        data_sink = 'forecast_data'
        test_config_yaml = os.path.join(self.test_data_path, 'test_configs/test_config_ssm.yml')
        module_config = cfg.configure(test_config_yaml, module)
        test_group = module_config['data_sinks'][data_sink]['group']
        forecast_expected_file = os.path.join(self.test_data_path, 'res/ssm_forecast_test.h5')
        with tempfile.TemporaryDirectory(prefix='temp') as test_out_dir:
            test_outcome_file = [test_out_dir, 'ssm_forecast_out.h5']
            module_config['data_sinks'][data_sink]['path'] = test_outcome_file
            temp_config_path = self.build_config_file_for_module(module_config, test_out_dir, module=module)

            # act
            # ---
            main_forecast(temp_config_path)

            # assert
            # ------
            # assert correct dimensions, coordinates, and data values
            outcome_path = os.path.join(test_outcome_file[0], test_outcome_file[1])
            event_rate_forecast = self.load_from_file(outcome_path, group=f'{test_group}/event_rate_forecast')
            nr_event_pmf = self.load_from_file(outcome_path, group=f'{test_group}/nr_event_pmf')
            full_forecast = self.load_from_file(outcome_path, group=f'{test_group}/forecast')

            expected_event_rate_forecast = self.load_from_file(forecast_expected_file, group='forecast/event_rate_forecast')
            expected_nr_event_pmf = self.load_from_file(forecast_expected_file, group='forecast/nr_event_pmf')
            expected_full_forecast = self.load_from_file(forecast_expected_file, group='forecast/forecast')

            xr.testing.assert_allclose(event_rate_forecast, expected_event_rate_forecast)
            xr.testing.assert_allclose(nr_event_pmf, expected_nr_event_pmf)
            xr.testing.assert_allclose(full_forecast, expected_full_forecast)

    @staticmethod
    def build_config_file_for_module(config: dict, out_dir: str, module: str, config_name: str = 'temp_config.yml') \
            -> str:
        """
        Write and save a new configuration .yaml file for a specific module.
        :param config: Dictionary with configuration for a specific module
        :param out_dir: Directory where the configuration files will be stored
        :param module: Name of the module
        :param config_name: Optional, name of the configuration file.
        :return: Returns the path to the new configuration file.
        """

        temp_module_config = {'modules': {module: config}}
        temp_yml_path = os.path.join(out_dir, config_name)
        with open(temp_yml_path, 'w') as f:
            yaml.dump(temp_module_config, f)

        return temp_yml_path


    @staticmethod
    def load_from_file(filepath, engine="h5netcdf", **kwargs):
        """
        Load an xarray object from a HDF5 dataset.

        Parameters
        ----------
        filepath : str
            The filepath to load from.
        engine : str, optional
            The engine to use to load. Defaults to 'h5netcdf'.

        Returns
        -------
        x : xarray.Dataset or xarray.DataArray
            The dataset or dataarray loaded from the file.
        """
        try:
            return xr.open_dataarray(filepath, engine=engine, **kwargs)
        except ValueError:
            return xr.open_dataset(filepath, engine=engine, **kwargs)