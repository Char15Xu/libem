import os
import json

import libem.prepare.datasets as datasets

description = "The failed benchmark cases from the 8 datasets."

# sample data:
# {"entity_1": {"name": "sharp over the counter microwave oven r1214ss", "description": "sharp over the counter microwave oven r1214ss 1.5 cubic foot capacity 1100 watts 24 automatic settings 2-color lighted lcd smart and easy sensor settings auto-touch control panel stainless steel finish", "price": 429.0}, 
# "entity_2": {"name": "sharp 1100 watt over the counter microwave", "description": null, "price": null}, 
# "pred": "yes", "label": 0, "model_output": "Yes", "tools_used": [], "latency": 0.42, "tokens": {"input_tokens": 148, "output_tokens": 1}}
def read(file, schema=True, **kwargs):
    with open(file) as f:
        for line in f:
            data = json.loads(line.strip())
            
            keep_null = 'keep_null' in kwargs and kwargs['keep_null']
            fields = kwargs['fields'] if 'fields' in kwargs else []
            price_diff = kwargs['price_diff'] if 'price_diff' in kwargs else False
            parsed_data = {'left': None, 'right': None, 'label': data.get('label', None)}
            left_values, right_values = {}, {}

            # clean the data
            if schema:
                price_l, price_r = 0, 0
                for key, value in data['entity_1'].items():
                    # Change null values to empty str
                    if not keep_null and value is None:
                        value = ''
                    if len(fields) == 0 or key in fields:
                        left_values[key] = value
                    if key == 'price' and value != "" and value != None:
                        price_l = value
                for key, value in data['entity_2'].items():
                    # Change null values to empty str
                    if not keep_null and value is None:
                        value = ''
                    if len(fields) == 0 or key in fields:
                        right_values[key] = value
                    if key == 'price' and value != "" and value != None:
                        price_r = value   
                
                if len(fields) > 0:
                    parsed_data['left'] = {field: left_values[field] for field in fields}
                    parsed_data['right'] = {field: right_values[field] for field in fields}
                else:
                    parsed_data['left'] = left_values
                    parsed_data['right'] = right_values
                            
                if price_diff:
                    if price_l == 0 or price_r == 0:
                        parsed_data['right']['price_difference'] = None if keep_null else ''
                    else:
                        difference = int(200 * abs(price_r - price_l) / (price_l + price_r))
                        parsed_data['right']['price_difference'] = str(difference) + '%'
                        
            else:
                for key, value in data['entity_1'].items():
                    # Change null values to empty str
                    if not keep_null and value is None:
                        value = ''
                    if len(fields) == 0 or key in fields:
                        left_values[key] = str(value)
                for key, value in data['entity_2'].items():
                    # Change null values to empty str
                    if not keep_null and value is None:
                        value = ''
                    if len(fields) == 0 or key in fields:
                        right_values[key] = str(value)
                            
                if len(fields) > 0:
                    parsed_data['left'] = ' '.join([left_values[field] for field in fields])
                    parsed_data['right'] = ' '.join([right_values[field] for field in fields])
                else:
                    parsed_data['left'] = ' '.join(left_values.values())
                    parsed_data['right'] = ' '.join(right_values.values())

            yield parsed_data


def read_test(schema=True, **kwargs):
    '''
    Yields processed records from the dataset one at a time.
    args:
        schema (bool): whether to include the schema or not
    kwargs:
        version (int): the version of the dataset to use, default to 0.
        keep_null (bool): if False, replace null values with empty str, else keep as 'None'.
        fields (list[str]): fields (and their order) to include in the output, 
                            empty to include all fields. Do not include _left/_right.
    '''
    version = int(kwargs['version']) if 'version' in kwargs else 0
    path = os.path.join(datasets.LIBEM_SAMPLE_DATA_PATH, "challenging")
    test_file = os.path.join(path, 'test.ndjson')
    
    return read(test_file, schema, **kwargs)


if __name__ == "__main__":
    import pprint

    pp = pprint.PrettyPrinter(sort_dicts=False)
    pp.pprint(next(read_test()))
    pp.pprint(next(read_test(schema=False)))
