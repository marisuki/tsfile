/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.tsfile.read;

import org.apache.tsfile.common.conf.TSFileConfig;
import org.apache.tsfile.common.conf.TSFileDescriptor;
import org.apache.tsfile.file.metadata.ChunkMetadata;
import org.apache.tsfile.file.metadata.IChunkMetadata;
import org.apache.tsfile.file.metadata.IDeviceID;
import org.apache.tsfile.utils.FileGenerator;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class MeasurementChunkMetadataListMapIteratorTest {

  private static final String FILE_PATH = FileGenerator.outputDataFile;
  private final TSFileConfig conf = TSFileDescriptor.getInstance().getConfig();

  private int maxDegreeOfIndexNode;

  @Before
  public void before() {
    maxDegreeOfIndexNode = conf.getMaxDegreeOfIndexNode();
    conf.setMaxDegreeOfIndexNode(3);
  }

  @After
  public void after() {
    FileGenerator.after();
    conf.setMaxDegreeOfIndexNode(maxDegreeOfIndexNode);
  }

  @Test
  public void test0() throws IOException {
    testCorrectness(1, 1);
    testSequentiality(1, 1);
  }

  @Test
  public void test1() throws IOException {
    testCorrectness(1, 10);
    testSequentiality(1, 10);
  }

  @Test
  public void test2() throws IOException {
    testCorrectness(2, 1);
    testSequentiality(2, 1);
  }

  @Test
  public void test3() throws IOException {
    testCorrectness(2, 2);
    testSequentiality(2, 2);
  }

  @Test
  public void test4() throws IOException {
    testCorrectness(2, 100);
    testSequentiality(2, 100);
  }

  @Test
  public void test5() throws IOException {
    testCorrectness(50, 2);
    testSequentiality(50, 2);
  }

  @Test
  public void test6() throws IOException {
    testCorrectness(50, 50);
    testSequentiality(50, 50);
  }

  @Test
  public void test7() throws IOException {
    testCorrectness(50, 100);
    testSequentiality(50, 100);
  }

  @Test
  public void test8() throws IOException {
    testCorrectness(33, 733);
    testSequentiality(33, 733);
  }

  @Test
  public void test9() throws IOException {
    testCorrectness(733, 33);
    testSequentiality(733, 33);
  }

  public void testCorrectness(int deviceNum, int measurementNum) throws IOException {
    FileGenerator.generateFile(10000, deviceNum, measurementNum);

    try (TsFileSequenceReader fileReader = new TsFileSequenceReader(FILE_PATH)) {
      Map<IDeviceID, List<String>> deviceMeasurementListMap = fileReader.getDeviceMeasurementsMap();

      List<IDeviceID> devices = fileReader.getAllDevices();

      Map<IDeviceID, Map<String, List<IChunkMetadata>>>
          expectedDeviceMeasurementChunkMetadataListMap = new HashMap<>();
      for (IDeviceID device : devices) {
        for (String measurement : deviceMeasurementListMap.get(device)) {
          expectedDeviceMeasurementChunkMetadataListMap
              .computeIfAbsent(device, d -> new HashMap<>())
              .computeIfAbsent(measurement, m -> new ArrayList<>())
              .addAll(fileReader.getChunkMetadataList(device, measurement, false));
        }
      }

      for (IDeviceID device : devices) {
        Map<String, List<IChunkMetadata>> expected =
            expectedDeviceMeasurementChunkMetadataListMap.get(device);

        Map<String, List<IChunkMetadata>> actual = new HashMap<>();
        Iterator<Map<String, List<ChunkMetadata>>> iterator =
            fileReader.getMeasurementChunkMetadataListMapIterator(device);
        while (iterator.hasNext()) {
          Map<String, List<ChunkMetadata>> next = iterator.next();
          for (Entry<String, List<ChunkMetadata>> entry : next.entrySet()) {
            actual.computeIfAbsent(entry.getKey(), m -> new ArrayList<>()).addAll(entry.getValue());
          }
        }

        checkCorrectness(expected, actual);
      }

      // test not exist device
      Iterator<Map<String, List<ChunkMetadata>>> iterator =
          fileReader.getMeasurementChunkMetadataListMapIterator(
              IDeviceID.Factory.DEFAULT_FACTORY.create("dd"));
      Assert.assertFalse(iterator.hasNext());
    }

    FileGenerator.after();
  }

  private void checkCorrectness(
      Map<String, List<IChunkMetadata>> expected, Map<String, List<IChunkMetadata>> actual) {
    Assert.assertEquals(expected.keySet(), actual.keySet());
    for (String measurement : expected.keySet()) {
      List<IChunkMetadata> expectedChunkMetadataList = expected.get(measurement);
      List<IChunkMetadata> actualChunkMetadataList = actual.get(measurement);
      Assert.assertEquals(expectedChunkMetadataList.size(), actualChunkMetadataList.size());
      final int size = expectedChunkMetadataList.size();
      for (int i = 0; i < size; ++i) {
        Assert.assertEquals(
            expectedChunkMetadataList.get(i).toString(), actualChunkMetadataList.get(i).toString());
      }
    }
  }

  public void testSequentiality(int deviceNum, int measurementNum) throws IOException {
    FileGenerator.generateFile(10000, deviceNum, measurementNum);

    try (TsFileSequenceReader fileReader = new TsFileSequenceReader(FILE_PATH)) {
      for (IDeviceID device : fileReader.getAllDevices()) {
        Iterator<Map<String, List<ChunkMetadata>>> iterator =
            fileReader.getMeasurementChunkMetadataListMapIterator(device);

        String lastMeasurement = null;
        while (iterator.hasNext()) {
          for (String measurement : iterator.next().keySet()) {
            if (lastMeasurement != null) {
              Assert.assertTrue(lastMeasurement.compareTo(measurement) < 0);
            }
            lastMeasurement = measurement;
          }
        }
      }
    }

    FileGenerator.after();
  }
}
