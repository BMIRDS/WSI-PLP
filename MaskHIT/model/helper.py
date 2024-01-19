import re


def parse_archs(arch):
    assert len(arch.split('_')) <= 2
    if '_' in arch:
        name, specs = arch.split('_')
        hh = int(re.findall('(?<=h)\\d+', specs)[0])
        try:
            ll = int(re.findall('(?<=l)\\d+', specs)[0])
        except:
            ll = 1
    else:
        name = arch
        hh, ll = 1, 1
    assert name in ['ap','attn','mhattn','deepattnmisl','vit']
    return {'name': name, 'h': hh, 'l': ll}
