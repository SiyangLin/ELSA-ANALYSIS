# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

onehotList_m=['couple', 'futype', 'dimar', 'wpaway', 'wpdes', 'wpcjob', 'wpslip', 
            'wpspd4', 'wpphi', 'iaspw', 'iapspw', 'wpslip', 'wpspd4', 'wpphi', 
            'iaspw', 'iapspw', 'ialiw', 'iadoc', 'iafcon', 'iaorgf', 'iafind', 
            'hocta', 'hodiff', 'hodifft', 'hodoc', 'horet', 'hocenp', 'hofpay', 
            'hogpay', 'hoepay', 'cflitsk', 'fffqethn', 'fffqcbth', 'fffqend', 
            'mmschs', 'screwh', 'diklivd', 'difjobd', 'phlegm', 'w5edqual', 'hotenu']

onehotList_2=['dhsameh', 'dhwork', 'disex', 'dhesmk', 'dheska', 'spcar', 'spcara',
              'spcarb', 'wpedc', 'wpes', 'wpjob', 'wpstj', 'wpemp', 'wpnic', 
              'wpnimw', 'wplnj', 'wpthl', 'wpcret', 'wpbus', 'wpspd','wpearly', 
              'wpexw', 'iawork', 'iaspen', 'iapspen', 'iafuel', 'iaden', 'iadem', 
              'iali', 'iasc', 'iaregp', 'iapar', 'iadebt', 'iaowe', 'howho', 
              'hotenun', 'homeal', 'hoctf', 'hocenh', 'hofsup', 'hoftog', 
              'hofd', 'hogd', 'hoed', 'hovnw', 'psceda', 'pscedb', 'pscedc', 
              'pscedd', 'pscede', 'pscedf', 'pscedg', 'pscedh', 'fqethnr', 
              'fqmqua', 'fffqwcul', 'fffqqual', 'mmavsp', 'mmwala', 'scmorea', 
              'scmoreb', 'scmorec', 'scmored', 'scptr', 'scchdm', 'scfam',
              'scfrd', 'scempl', 'scal7a', 'mother', 'father', 'wpmoj', 'scchd']

continuousAttr = ['diagr', 'dignmy', 'disib', 'sptraa','wpvw', 'spsare', 'spsacc', 
                  'splift', 'wpactw', 'wpvw', 'wpnoe', 'wpjact', 'wpcjob', 
                  'wphjob', 'wpwlym', 'wphly', 'wpaotp', 'wpthp', 'wppaya',
                  'wpmpah', 'wpmala', 'wpmsah', 'wpmsh', 'iasinc', 'iapam',
                  'iappam', 'iappen', 'iappmo', 'iasava', 'iasint', 'ianpb', 
                  'ianpbp', 'iacisa', 'iaisad', 'iasss', 'iasssi', 'ialiya', 
                  'ialipa', 'iainta', 'hoask', 'homove', 'hosellp', 'hoctn',
                  'hointa', 'horoom', 'hoold', 'hohv01', 'hooutf', 'hocl', 
                  'holeis', 'hotran', 'hotrangc', 'hotranoc', 'hotranch', 'dignmyd',
                  'hotrannr', 'hofdm', 'hogdm', 'hoedm', 'hoveh', 'erfvoft', 'erivoft',
                  'ervola', 'ervolb', 'scorgn', 'scedcp', 'scedpo', 'scedcs', 
                  'scedsu', 'scedsc', 'scedgp', 'scedch', 'scedde', 'scedop',
                  'scedho', 'scacta', 'scactb', 'scactc', 'scactd', 'sctvwkd', 
                  'sctvwke', 'sclifea', 'sclifeb', 'sclifec', 'sclifed', 'sclifee',
                  'scdca', 'scdcc', 'scdcd', 'scdce', 'scdcg', 'scfeela', 
                  'scfeelb', 'scfeelc', 'scfeeld', 'scfeele',
                  'scqola', 'scqolb', 'scqolc', 'scqold', 'scqole', 'scqolf', 
                  'scqolg', 'scqolh', 'scqoli', 'scqolj', 'scqolk', 'scqoll',
                  'scqolm', 'scqoln', 'scqolo', 'scqolp', 'scqolq', 'scqolr',
                  'scqols', 
                  'scfede', 'scfeen', 'scfeac', 'scfepr', 'scfeint',
                  'scfeha', 'scfeat', 'scfeco', 'scfeins', 'scfeho', 'scfeal',
                  'scfeca', 'scfeex', 
                  'scdeou', 'scdehe', 'scdemo', 'scdeor', 'scdefr', 'scdewa',
                  'scdewo', 'scdere', 'scdeli', 'scdeca', 'scdene', 'scdecr',
                  'scdeha', 'scdeim', 'scdesof', 'scdecal', 'scdein', 'scdecu',
                  'scdeac', 'scdecar', 'scdebr', 'scdesy', 'scdeta', 'scdeso',
                  'scdead', 'scdeth',
                  'scptra', 'scptrb', 'scptrc', 'scptrd', 'scptre', 'scptrf', 
                  'scptrx', 'scptrg',
                  'scchda', 'scchdb', 'scchdc', 'scchde', 'scchdf', 'scchdg', 
                  'scchdh', 'scchdi', 'scchdm', 'scchdx', 'scchdd',
                  'scfama', 'scfamb', 'scfamc', 'scfamd', 'scfame', 'scfamf', 
                  'scfamg', 'scfamh', 'scfami', 'scfamm', 'scfamx',
                  'scfrda', 'scfrdb', 'scfrdc', 'scfrde', 'scfrdf', 'scfrdg', 
                  'scfrdh', 'scfrdi', 'scfrdm', 'scfrdx', 'scfrdd',
                  'scdtdre', 'scdtdst', 'scdtdcl', 'scdtdha', 'scdtddr',
                  'screlof', 'screlfa', 'screlfa', 'screlpr', 'screlme', 
                  'screlac', 'screlim', 
                  'scworka', 'scworkb', 'scworkc', 'scworkd', 'scworke', 
                  'scworkf', 'scworkg', 'scworkh', 'scworki', 'scworkj', 
                  'scworkk', 'scworkl',
                  'scrtage', 'sclddr', 'scveg', 'scfru', 'scako', 'scal7b',
                  'scdrspi', 'scdrwin', 'scdrpin', 'alltotch', 'mthagd', 'fthagd',
                  'disibd', 
                  'palevel', 'breths', 'cfmersp', 'cfprmem', 'cfmeind', 'cfexind',
                  'cfind', 'cfaccur',
                  'cfanig', 'cfrecal', 'fstgs_tm', 'gtspd_wk', 'gtspd_mn',
                  'gtspd_mng',
                  'CASP19', 'CASPCTL', 'CASPAUT', 'CASPPLE', 'CASPSR',
                  'organis']

#note={'splift':6.0}

def preProcess(df):
    def fillnan(x):
        t=x.copy()
        t[x<0]=np.nan
        return t
    df[onehotList_2] = df[onehotList_2].apply(fillnan)
    df[onehotList_m] = df[onehotList_m].apply(fillnan)
    df[df<0]=0
#    df[continuousAttr]=df[continuousAttr].apply(lambda x:(x-x.mean())/x.std())
#    df[continuousAttr]=df[continuousAttr].apply(lambda x:(x-x.min()))
    df[continuousAttr]=df[continuousAttr].apply(lambda x:(x-x.min())/(x.max()-x.min()))
    df=pd.get_dummies(df, columns=onehotList_m)
    df=pd.get_dummies(df, columns=onehotList_2)
    return df
