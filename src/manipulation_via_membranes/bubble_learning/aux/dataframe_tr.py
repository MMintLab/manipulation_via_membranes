from pandas import DataFrame


class SplitDataFramesTr(object):

    def __init__(self, keys_to_tr=None):
        self.keys_to_tr = keys_to_tr

    def __call__(self, sample):
        if self.keys_to_tr is None:
            # transform all the DataFrames
            all_keys = list(sample.keys())
            for key in all_keys:
                v = sample[key]
                if type(v) is DataFrame:
                    # Split the dataframe into column and data
                    sample = self._tr(sample, key)
        else:
            for key in self.keys_to_tr:
                if key in sample:
                    sample = self._tr(sample, key)
        return sample

    def inverse(self, sample):
        if self.keys_to_tr is None:
            # search for key pairs with key_columns and key_data
            sample_keys = sample.keys()
            column_keys = [k for k in sample_keys if '_columns' in k]
            data_keys = [k for k in sample_keys if '_data' in k]
            common_keys = []
            for col_k in column_keys:
                common_key = col_k.split('_columns')[0]
                data_key = '{}_data'.format(common_key)
                if data_key in data_keys:
                    common_keys.append(common_key)
            for cm_k in common_keys:
                sample = self._tr_inv(sample, cm_k)
        else:
            for key in self.keys_to_tr:
                sample = self._tr_inv(sample, key)

        return sample

    def _tr(self, sample, key):
        v = sample[key]
        if type(v) is DataFrame:
            sample['{}_columns'.format(key)] = v.columns.values
            sample['{}_data'.format(key)] = v.values
            # remove old key-value pair
            sample.pop(key)
        return sample

    def _tr_inv(self, sample, key):
        v = sample[key]
        column_key = '{}_columns'.format(key)
        data_key = '{}_data'.format(key)

        sample[key] = DataFrame(sample[data_key], columns=sample[column_key])
        # remove transformed pairs
        sample.pop(column_key)
        sample.pop(data_key)

        return sample

