class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def float2str(num):
    nums = str(num).split('.')
    return nums[0] + '.' + nums[1][0:2]
