import logging
import atexit
import boto3
import watchtower
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LogManager:
    """CloudWatch logging configuration manager."""

    @staticmethod
    def setup_logging(config: Dict[str, Any], secret: Dict[str, Any]) -> None:
        """Initialize CloudWatch logging with instance tracking."""
        try:
            logger = logging.getLogger(__name__)
            logger.info("Setting up CloudWatch logging")

            # Reduce noise from asyncio
            logging.getLogger("asyncio").setLevel(logging.WARNING)

            # Initialize CloudWatch client
            cloudwatch_client = boto3.client(
                "logs",
                aws_access_key_id=secret["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"],
                region_name=config["AWS_REGION"],
            )

            # Get EC2 instance information
            logger.debug("Retrieving EC2 instance ID")
            ec2_client = boto3.Session().resource(
                "ec2", region_name=config["AWS_REGION"]
            )
            instance_id = ec2_client.meta.client.describe_instances()["Reservations"][
                0
            ]["Instances"][0]["InstanceId"]

            # Configure CloudWatch handler
            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group=config["CLOUDWATCH_LOG_GROUP"],
                stream_name=instance_id,
                boto3_client=cloudwatch_client,
                use_queues=False,
            )

            # Setup logging configuration
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[cloudwatch_handler],
            )

            # Ensure logs are flushed on shutdown
            atexit.register(lambda: cloudwatch_handler.flush())

            logger.info(f"CloudWatch logging setup complete for instance {instance_id}")

        except Exception as e:
            logger.critical(f"Failed to setup CloudWatch logging: {e}")
            raise 