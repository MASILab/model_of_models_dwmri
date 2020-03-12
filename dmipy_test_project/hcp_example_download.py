from dmipy.hcp_interface import downloader_aws

public_aws_key = 'AKIAXO65CT57D3ITH2XM'
secret_aws_key = 'XBx80UPkphT4bgkArHN6QY7GX+cfUJcep/di60Jv'

hcp_interface = downloader_aws.HCPInterface(
    your_aws_public_key=public_aws_key,
    your_aws_secret_key=secret_aws_key)

hcp_interface.download_and_prepare_dmipy_example_dataset()