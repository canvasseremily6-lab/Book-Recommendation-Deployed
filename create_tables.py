import boto3
from botocore.exceptions import ClientError

dynamodb = boto3.resource('dynamodb', region_name = 'us-east-1')

# creates both tables for prediction logs and recommendations
def create_table_if_not_exists(table_name,key_name):
    try: 
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'request_id', 'KeyType': 'HASH'},
            ],
            AttributeDefinitions=[
                {'AttributeName': 'request_id', 'AttributeType': 'S'},
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        dynamodb.Table(table_name).wait_until_exists()
        print(f'{table_name} created and active')
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceInUseException':
            print(f'{table_name} already exists')
        else: 
            raise 

create_table_if_not_exists('prediction_logs', 'request_id')
create_table_if_not_exists('recommendation_cache', 'title_key')

dynamodb.Table('prediction_logs').wait_until_exists()
dynamodb.Table('recommendation_cache').wait_until_exists()

print('Both tables are active')