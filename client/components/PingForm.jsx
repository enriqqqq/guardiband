import Paho from 'paho-mqtt';

const PingForm = ({ mqttClient, handleClose }) => {
    // give form to send mqtt message
    const [pingValue, setPingValue] = useState('Ping');

    const sendMessage = () => {
        if(mqttClient && mqttClient.isConnected()) {
            const message = new Paho.Message(JSON.stringify({
                type: 'ping',
                message: pingValue
            }));
            message.destinationName = 'guardiband/1A/message';
            console.log('Sending message:', message);
            mqttClient.send(message);
            handleClose();
        }
    }

    const handlePingValueChange = (value) => {
        // limit to 15 characters
        if (value.length <= 15) {
            setPingValue(value);
        }
    }

    return (
        <>
            <View className="absolute bg-black opacity-50 w-full h-full"></View>
            <View className="flex-1 justify-center items-center">
                <View className="bg-white p-5 rounded">
                    <Text className="text-xl font-bold">Send a message</Text>
                    <View className="mt-1 mb-3">
                        <Text>Message</Text>
                        <TextInput className="bg-white border rounded p-2" value={pingValue} onChangeText={handlePingValueChange}></TextInput>
                        <Text className="text-xs">{pingValue.length}/15</Text>
                    </View>
                    <View className="gap-1">
                        <Button
                            title="Send"
                            onPress={sendMessage}
                        ></Button>
                        <Button
                            title="Close"
                            onPress={handleClose}
                            color={'red'}
                        ></Button>
                    </View>
                </View>
            </View>
        </>
    )
}

export default PingForm;