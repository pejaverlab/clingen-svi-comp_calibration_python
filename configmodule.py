import configparser
import os


class ConfigModule:
    
    def __init__(self):
        self.alpha = None
        self.B = None
        self.discountonesided = None
        self.windowclinvarpoints = None
        self.windowgnomadfraction = None
        self.emulate_tavtigian = None
        self.emulate_pejaver = None
        self.gaussian_smoothing = None
        self.data_smoothing = None
        self.cfg = None

    def load_config(self, filepath):

        cfg = configparser.ConfigParser()
        cfg.read(filepath)
        cfg.sections()
        self.cfg = cfg
        
        self.B = int(cfg['tuningparameters']['B'])
        self.discountonesided = float(cfg['tuningparameters']['discountonesided'])
        self.windowclinvarpoints = int(cfg['tuningparameters']['windowclinvarpoints'])

        self.emulate_tavtigian = cfg['priorinfo'].getboolean('emulate_tavtigian')
        self.emulate_pejaver = cfg['priorinfo'].getboolean('emulate_pejaver')
        assert not (self.emulate_tavtigian and self.emulate_pejaver)

        if not (self.emulate_tavtigian or self.emulate_pejaver):
            self.alpha = float(cfg['priorinfo']['alpha'])
            assert self.alpha is not None
            

        self.data_smoothing = cfg['smoothing'].getboolean('data_smoothing')
        if self.data_smoothing:
            self.windowgnomadfraction = float(cfg['smoothing']['windowgnomadfraction'])
            assert self.windowgnomadfraction is not None
        self.gaussian_smoothing = cfg['smoothing'].getboolean('gaussian_smoothing')


    def save_config(self, outdir):
        with open(os.path.join(outdir, "config.ini"), 'w') as configfile:    # save
            self.cfg.write(configfile)
