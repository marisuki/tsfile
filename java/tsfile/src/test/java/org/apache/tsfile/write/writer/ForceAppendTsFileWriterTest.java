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
package org.apache.tsfile.write.writer;

import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.file.metadata.enums.TSEncoding;
import org.apache.tsfile.fileSystem.FSFactoryProducer;
import org.apache.tsfile.fileSystem.fsFactory.FSFactory;
import org.apache.tsfile.read.TsFileReader;
import org.apache.tsfile.read.TsFileSequenceReader;
import org.apache.tsfile.read.common.Path;
import org.apache.tsfile.read.common.RowRecord;
import org.apache.tsfile.read.expression.QueryExpression;
import org.apache.tsfile.read.query.dataset.QueryDataSet;
import org.apache.tsfile.utils.TsFileGeneratorForTest;
import org.apache.tsfile.write.TsFileWriter;
import org.apache.tsfile.write.record.TSRecord;
import org.apache.tsfile.write.record.datapoint.FloatDataPoint;
import org.apache.tsfile.write.schema.MeasurementSchema;

import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class ForceAppendTsFileWriterTest {
  private static final String FILE_NAME =
      TsFileGeneratorForTest.getTestTsFilePath("root.sg1", 0, 0, 1);
  private static FSFactory fsFactory = FSFactoryProducer.getFSFactory();

  @Test
  public void test() throws Exception {
    File file = fsFactory.getFile(FILE_NAME);
    if (file.exists()) {
      fail("Do not know why the file exists...." + file.getAbsolutePath());
    }
    System.out.println(file.getAbsolutePath());
    if (!file.getParentFile().exists()) {
      Assert.assertTrue(file.getParentFile().mkdirs());
    }
    if (!file.getParentFile().isDirectory()) {
      fail("folder is not a directory...." + file.getParentFile().getAbsolutePath());
    }

    TsFileWriter writer = new TsFileWriter(file);
    writer.registerTimeseries(
        new Path("d1"), new MeasurementSchema("s1", TSDataType.FLOAT, TSEncoding.RLE));
    writer.registerTimeseries(
        new Path("d1"), new MeasurementSchema("s2", TSDataType.FLOAT, TSEncoding.RLE));
    writer.writeRecord(
        new TSRecord("d1", 1)
            .addTuple(new FloatDataPoint("s1", 5))
            .addTuple(new FloatDataPoint("s2", 4)));
    writer.writeRecord(
        new TSRecord("d1", 2)
            .addTuple(new FloatDataPoint("s1", 5))
            .addTuple(new FloatDataPoint("s2", 4)));
    writer.flush();

    long firstMetadataPosition = writer.getIOWriter().getPos();
    writer.close();
    ForceAppendTsFileWriter fwriter = new ForceAppendTsFileWriter(file);
    assertEquals(firstMetadataPosition, fwriter.getTruncatePosition());
    fwriter.doTruncate();

    // write more data into this TsFile
    writer = new TsFileWriter(fwriter);
    writer.registerTimeseries(
        new Path("d1"), new MeasurementSchema("s1", TSDataType.FLOAT, TSEncoding.RLE));
    writer.registerTimeseries(
        new Path("d1"), new MeasurementSchema("s2", TSDataType.FLOAT, TSEncoding.RLE));
    writer.writeRecord(
        new TSRecord("d1", 3)
            .addTuple(new FloatDataPoint("s1", 5))
            .addTuple(new FloatDataPoint("s2", 4)));
    writer.close();
    TsFileReader tsFileReader = new TsFileReader(new TsFileSequenceReader(file.getPath()));
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s1", true));
    pathList.add(new Path("d1", "s2", true));
    QueryExpression queryExpression = QueryExpression.create(pathList, null);
    QueryDataSet dataSet = tsFileReader.query(queryExpression);
    RowRecord record = dataSet.next();
    assertEquals(1, record.getTimestamp());
    assertEquals(5.0f, record.getFields().get(0).getFloatV(), 0.001);
    assertEquals(4.0f, record.getFields().get(1).getFloatV(), 0.001);
    record = dataSet.next();
    assertEquals(2, record.getTimestamp());
    assertEquals(5.0f, record.getFields().get(0).getFloatV(), 0.001);
    assertEquals(4.0f, record.getFields().get(1).getFloatV(), 0.001);
    record = dataSet.next();
    assertEquals(3, record.getTimestamp());
    assertEquals(5.0f, record.getFields().get(0).getFloatV(), 0.001);
    assertEquals(4.0f, record.getFields().get(1).getFloatV(), 0.001);
    tsFileReader.close();
    assertFalse(dataSet.hasNext());

    assertTrue(file.delete());
  }
}
