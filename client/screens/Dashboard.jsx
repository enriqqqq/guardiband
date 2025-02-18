import { View, Text, Button, Modal, Alert, TextInput } from 'react-native';
import MapView, { Marker, Polyline } from 'react-native-maps';
import openMap from 'react-native-open-maps';
import Icon from 'react-native-vector-icons/FontAwesome';
import Paho from 'paho-mqtt';
import { useState, useEffect, useRef } from 'react';

const Dashboard = () => {
    const [mqttData, setMqttData] = useState([
        {
            temp: "0",
            spo2: "0",
            heartRate: "0",
            lat: "-6.2013465",
            lon: "106.7814731",
            firstReadGPS: false
        }
    ]);

    const [coordinates, setCoordinates] = useState([
        {
            latitude: -6.2013465,
            longitude: 106.7814731
        }
    ]);

    const [showPingForm, setShowPingForm] = useState(false);
    const [mapReady, setMapReady] = useState(false);
    const clientRef = useRef(null);
    
    useEffect(() => {
        console.log('useEffect for MQTT connection');
        // use 'test.mosquitto.org' as a test broker
        // const client = new Paho.Client('test.mosquitto.org', 8080, '1A-guardian');
        // const client = new Paho.Client('broker.hivemq.com', 8884, '1A-guardian');
        // const client = new Paho.Client('broker.emqx.io', 8083, '1A-guardian');
        const client = new Paho.Client('id1f113f.ala.asia-southeast1.emqxsl.com', 8084, '1A-guardian');

        // Connect the client
        client.connect({
            useSSL: true,
            userName: 'guardiband-mqtt',
            password: 'guardiband',
            onSuccess: () => {
                console.log('Connected');
                client.subscribe('guardiband/1A/data');
            },
            onFailure: (error) => {
                console.log('Connection failed:', error);
            },
        });

        client.onConnectionLost = (responseObject) => {
            console.log('Connection lost:', responseObject.errorMessage);
        };

        // handle incoming messages
        client.onMessageArrived = (message) => {
            console.log('Message arrived:\n', message.payloadString, 'on topic', message.destinationName);
            
            // convert the payload to JSON
            const payload = JSON.parse(message.payloadString);

            // check which type of message it is
            const type = payload.type;

            switch (type) {
                case "data":
                    // limit the data to 10, store mqtt data
                    setMqttData((prevData) => {
                        // check if the data is already 10, if so, remove the first element
                        if (prevData.length >= 10) {
                            return [...prevData.slice(1), payload];
                        }
                        return [...prevData, payload];
                    });

                    // store the coordinates logic
                    setCoordinates((prevCoords) => {
                        const lastCoord = prevCoords[prevCoords.length - 1];
                        const newCoord = { latitude: parseFloat(payload.lat), longitude: parseFloat(payload.lon) };
                        const distance = Math.sqrt(Math.pow(newCoord.latitude - lastCoord.latitude, 2) + Math.pow(newCoord.longitude - lastCoord.longitude, 2));
                        console.log('Distance:', distance);
                        console.log('Last coord:', lastCoord);
                        
                        // if firstReadGPS is false don't add new coordinate
                        if(!payload.firstReadGPS) {
                            return prevCoords;
                        }


                        // if the distance is more than 5m, add the new coordinate
                        if(distance > 0.00005) {
                            const updatedCoords = [...prevCoords, newCoord];
                            console.log('Updated coords:', updatedCoords);
                            // limit the coordinates to 5
                            if(updatedCoords.length > 5) {
                                updatedCoords.shift();
                            }
                            return updatedCoords;
                        }

                        return prevCoords;
                    });

                    break;
                case "alert":
                    console.log(payload.message);
                    Alert.alert('Alert Title', payload.message);
                    break;
            }
        };

        // Save the client to the ref
        clientRef.current = client;

        return () => {
            console.log('Disconnecting client');
            client.disconnect();
        };
    }, []);

    return (
        <View className="flex-1 bg-gray-100 p-9">
            <Modal visible={showPingForm} transparent={true} onRequestClose={() => setShowPingForm(false)}>
                <PingForm mqttClient={clientRef.current} handleClose={() => setShowPingForm(false)}/>
            </Modal>

            <Text className="text-2xl font-bold mt-9">Dashboard</Text>
            <View className="flex-col mt-5 flex-1">

                <View className="flex-row gap-4">
                    {/* This is for temperature */}
                    <View className="bg-white flex-1 rounded p-3 shadow-md flex-row">
                        <View className="items-center justify-center bg-blue-50 flex-1 w-auto">
                            <Icon name="thermometer" size={32} color="#900"/>
                        </View>
                        <View className="bg-red-50 flex-grow justify-center items-center">
                            <Text className="text-lg font-bold">Temp</Text>
                            <Text>{`${mqttData[mqttData.length - 1].temp} °C`}</Text>
                        </View>
                    </View>

                    {/* This is for SpO2 */}
                    <View className="bg-white flex-1 rounded p-3 shadow-md flex-row">
                        <View className="items-center justify-center bg-blue-50 flex-1 w-auto">
                            <Icon name="percent" size={32} color="#900"/>
                        </View>
                        <View className="bg-red-50 flex-grow justify-center items-center">
                            <Text className="text-lg font-bold">SpO2</Text>
                            <Text>{`${mqttData[mqttData.length - 1].spo2} %`}</Text>
                        </View>
                    </View>
                </View>

                <View className="bg-white rounded p-3 mt-2 flex-row">
                    <View className="items-center justify-center bg-blue-50 w-auto p-2">
                        <Icon name="heartbeat" size={32} color="#900"/>
                    </View>
                    <View className="bg-red-50 flex-grow justify-center items-center">
                        <Text className="text-lg font-bold">Heart Rate</Text>
                        <Text>{`${mqttData[mqttData.length - 1].heartRate} BPM`}</Text>
                    </View>
                </View>

                <View className="mt-5 flex-1">
                    <Text className="text-lg font-medium">Location</Text>
                    <View className="flex-1">

                        <MapView
                            onMapReady={() => setMapReady(true)}
                            initialRegion={{
                                latitude: -6.2013465,
                                longitude: 106.7814731,
                                latitudeDelta: 0.005,
                                longitudeDelta: 0.005,
                            }}
                            style={{ marginTop: 10, marginBottom: 20, borderRadius: 10, flex:1 }}
                        >
                            {
                                mapReady && (
                                    <>
                                        <Marker
                                            coordinate={{
                                                latitude: coordinates[coordinates.length - 1].latitude,
                                                longitude: coordinates[coordinates.length - 1].longitude
                                            }}
                                            title="My Location"
                                            description="Here I am"
                                        />
                                        <Polyline
                                            coordinates={coordinates}
                                            strokeColor="blue"
                                            strokeWidth={2}
                                        />
                                    </>
                                )
                            }
                        </MapView>

                    </View>
                    <View className="gap-3">
                        <Button
                            title="Open Map"
                            onPress={() => openMap({
                                latitude: coordinates[coordinates.length - 1].latitude,
                                longitude: coordinates[coordinates.length - 1].longitude,
                                zoom: 20
                            })}
                        ></Button>
                        <Button
                            title="Send Message"
                            onPress={() => setShowPingForm(true)}
                        ></Button>
                    </View>
                </View>

            </View>
        </View>
    );
}

const PingForm = ({ mqttClient, handleClose }) => {
    // give form to send mqtt message
    const [pingValue, setPingValue] = useState('Ping');

    const sendMessage = () => {
        if(mqttClient && mqttClient.isConnected()) {
            const message = new Paho.Message(pingValue);
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

export default Dashboard;